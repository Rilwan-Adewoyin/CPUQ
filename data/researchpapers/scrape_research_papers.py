from __future__ import annotations
import os
import sys
sys.path.append(os.getcwd())

from io import BytesIO

from argparse import ArgumentParser

import yaml
import multiprocessing as mp
import itertools

import time

import aiohttp
from aiohttp import  ClientError

import asyncio

# import logging
# logger = logging.getLogger('logger')
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))

from typing import List, Sequence
# from bs4 import BeautifulSoup
import requests

import fitz
import csv 
import wget

import gzip as gz
from fake_headers import Headers
from requests_html import HTMLSession, AsyncHTMLSession

import json
import numpy as np

from pdfminer.high_level import extract_pages, extract_text
import pdftotext
from prompt_engineering.my_logger import setup_logging_scrape_rps
logger = None
"""
    This script scrapes research papers from Google Scholar.
    Then converts the research papers to a .txt document format.
    Then performs preprocessing.
    Then saves in pkl format.
    #TODO: Ensure that no duplicates pdfs are downloaded

    # We use search terms related to each of the broad budget categories

"""

conccurent_tasks = 1
sem = asyncio.Semaphore(conccurent_tasks)  # Limiting to 10 concurrent tasks.

def main(
    downloads_per_search_term:int,
    min_citations:int,
    source:str,
    mp_count=4,
    pdf_parser:str='pdfminer',
    pdfs_downloaded:bool=False,
    debugging:bool=False):

    search_terms = yaml.safe_load(open('./data/researchpapers/search_terms.yaml','r'))
    
    
    global logger 
    logger =  setup_logging_scrape_rps(debugging=debugging)

    if debugging:
        downloads_per_search_term = 1
        search_terms = search_terms[:1]

    if not pdfs_downloaded:
        # scrape pdfs
        logger.info("Scraping pdfs")
        li_li_pdf_title_authors = asyncio.run( scrape_pdfs( search_terms, downloads_per_search_term, min_citations, source ) )
        logger.info("Finished Scraping pdfs")

        # remove duplicates from li_li_pdf_title_authors by checking for duplicated titles
        li_li_pdf_title_authors = remove_duplicates(li_li_pdf_title_authors)

        # save pdfs to file
        logger.info("Saving pdfs to file")
        with mp.Pool(mp_count) as p:
            res = p.starmap(save_pdfs,  list(zip(search_terms, itertools.count(), li_li_pdf_title_authors))  )
    
    else:
        # load pdfs from file
        logger.info("Loading pdfs from file")
        li_li_pdf_title_authors = load_pdfs(search_terms)
    
    # extract texts
    logger.info("Covnerting pdf to txt")
    li_pdfs = [pdf for pdf, title, author in sum(li_li_pdf_title_authors, []) ] # all pdfs flattened
    with mp.Pool(mp_count) as p:

        if pdf_parser=='pdfminer':
            gen_texts = p.imap(extract_text_pdfminer, li_pdfs, chunksize=1  )
        elif pdf_parser=='pdftotext':
            gen_texts = p.imap(extract_text_pdftotext, li_pdfs, chunksize=1  )
        elif pdf_parser=='fitz':
            gen_texts = p.imap(extract_text_fitz, li_pdfs, chunksize=1  )
        else:
            raise ValueError(f"pdf_parser: {pdf_parser} not supported")

        # Replacing pdfs in li_li_pdf_title_authors with processed text
        li_li_txt_title_author = li_li_pdf_title_authors

        for idx_searchterm in range(len(li_li_pdf_title_authors)):
            
            for idx_rsearchpaper in range(len(li_li_pdf_title_authors[idx_searchterm])):
                
                new_vals = ( next( gen_texts ), 
                             *li_li_txt_title_author[idx_searchterm][idx_rsearchpaper][1:]
                             )

                li_li_txt_title_author[idx_searchterm][idx_rsearchpaper] =  new_vals
            
    
    # Saving Texts to file
    logger.info("Saving txt to file")
    with mp.Pool(mp_count) as p:
        res = p.starmap(save_text, list(zip(search_terms, itertools.count(), li_li_txt_title_author)) )
    
    logger.info("Script Finished")
    return None
   
async def scrape_pdfs( search_terms, downloads_per_search_term, min_citations, source:str ) -> list[ list[tuple[bytes|int,str,str]]  ]:
    
    async with aiohttp.ClientSession(headers={'User-Agent':'Mozilla/5.0' } ) as session:
    # ,connector=TCPConnector(ssl=False) ) as session:
    # async with aiohttp.ClientSession() as session:
        logger.info("Started aiohttp Session")
        

        li_tasks = [None]*len(search_terms)
        
        if source == 'google_scholar':
            scrape_func =  scrape_pdfs_google_scholar
        elif source == 'semantic_scholar':
            scrape_func = get_pdfs_semantic_scholar_api
        else:
            raise ValueError(f"source: {source} not supported")

        for idx, search_term in enumerate(search_terms):
            # li_tasks[idx] = asyncio.create_task(scrape_pdfs_google_scholar(session, search_term, downloads_per_search_term, min_citations))            
            li_tasks[idx] = scrape_func(session, search_term, downloads_per_search_term, min_citations, proc_idx_totalcount = (idx, len(search_terms)), logger=logger )
                    
        li_pdf_title_author = await asyncio.gather(*li_tasks)

    return li_pdf_title_author

async def scrape_pdfs_google_scholar(session, search_term:str, downloads_per_search_term:int, min_citations:int) -> list[tuple[str, bytes]]:
    
    # Helper class to generate headers for requests
    # NOTE: TO avoid brotli encoded responses, ".update({'accept-encoding': 'gzip, deflate, utf-8'})":" is appended to the generate output
    headers = Headers(os='win', headers=True)

    docs_per_url = 10
    
    li_pdf = []
    li_title = []
    li_author = [] #NOTE: currently author not scraped

    headers1 = headers.generate().update({'accept-encoding': 'gzip, deflate, utf-8'})
    headers2 = headers.generate().update({'accept-encoding': 'gzip, deflate, utf-8'})

    # open webpage
    for idx in itertools.count():
    
        url = f"https://scholar.google.com/scholar?start={docs_per_url*idx}&q={search_term.replace(' ','+')}&hl=en"

        async with session.get(url, headers=headers1 ) as resp:
            time.sleep(5.5)

            # if no more pages then break
            if resp.status != 200:
                break

            # convert to beautilful soup tree
            text = await resp.read()
            soup = BeautifulSoup(text, 'lxml')

        # searching and extracting pdf links

        ## getting the html divisions representing a single research paper
        tags_research_papers = soup.find_all( 'div', {'class','gs_r gs_or gs_scl'}, recursive=True )

        ## filtering for html div tag (for research papers) that have a link to a pdf file
        for idx1 in reversed(range(len(tags_research_papers))):
            
            tag = tags_research_papers[idx1]
            res = tag.find(href= lambda txt: (txt is not None) and txt[-4:]=='.pdf' )

            if res is None:
                tags_research_papers.pop(idx1)

        ## filtering for html div representations of research papers that have at least min_citations citations
        for idx2 in reversed(range(len(tags_research_papers))):
            tag = tags_research_papers[idx2]

            # research paper has a child tag with text 'Cited by N' where N at least min citations
            res = tag.find( string = lambda txt:  ('Cited by' in txt) and (min_citations<=int( txt.split(' ')[-1]) ) )

            if res is None:
                tags_research_papers.pop(idx2)

        ### titles are the text in <a> tags that have a parent tag with class name 'gs_rt'
        titles = [ ''.join(tag.find( class_='gs_rt' ).strings) for tag in tags_research_papers]   

        ## extracting pdf urls
        urls = [ tag.find(href= lambda txt: (txt is not None) and txt[-4:]=='.pdf' ).attrs['href'] for tag in tags_research_papers ]
        

        pdfs = []
        for url in urls:
            try:
                time.sleep(5.0)
                pdf = await (await session.get(url, cookies=resp.cookies,
                            headers=headers2,
                            #  verify_ssl=False,
                            #  ssl=False,
                            # ssl_context = ssl_context
                            )).content.read()
            except (ClientError) as e:
                pdf = "NAN"
            pdfs.append(pdf)


        # Note: Cloudfare blocking disables all pdfs linked via ResearchGate, remove downloads that were content blocked
        _ =[(pdf,title) for pdf,title in zip(pdfs,titles) if pdf[:4]==b'%PDF']
        if len(_)>0:
            pdfs,titles =  list( zip(*_))
        else:
            pdfs,titles = [],[]

        li_title.extend(titles)
        li_pdf.extend(pdfs)
        
        # break pdf urls collected exceeds min citations
        if len(li_pdf)>=downloads_per_search_term:
            break
            
    # filter urls on existence of download link, and min citations, link containing text [PDF]
    li_pdf = li_pdf[:downloads_per_search_term]
    li_title = li_title[:downloads_per_search_term]
    li_author = ['']*len(li_pdf)

    
    outp = list( zip(li_pdf, li_title, li_author))
    return outp

async def get_pdfs_semantic_scholar_api(session, search_term:str, downloads_per_search_term:int, min_citations:int, proc_idx_totalcount:tuple[int, int], logger=None) -> list[tuple[str, bytes]]:
    # rate limit of 100 requests per 5 minutes, 1 request per 3 seconds
    
    async with sem:

        # Helper class to generate headers for requests
        # NOTE: TO avoid brotli encoded responses, ".update({'accept-encoding': 'gzip, deflate, utf-8'})":" is appended to the generate output
        headers = Headers(os='win', headers=True)
        
        li_pdf = []
        li_title = []
        li_author = []

        papers_per_query = 100
        # open webpage

        start_time = time.time()

        for idx in itertools.count(start=0):
                
            url_base = "https://api.semanticscholar.org/graph/v1/paper/search?"
            url_query = f"query={search_term.replace(' ','+')}"
            url_filters = "openAccessPdf&year=2000-2021"
            url_fields = "fields=title,authors,citationCount,openAccessPdf"
            url_paper_count = f"offset={str(idx*papers_per_query)}&limit={str(papers_per_query)}"
            url_lang = "lang=en"
            url_fieldsofstudy = 'fieldsOfStudy='+','.join(['Sociology','Political Science','Economics','Law','Education'])
            
            url = url_base+'&'.join([url_query, url_filters, url_fields, url_paper_count, url_lang, url_fieldsofstudy])

            headers1 = {
                "Accept": "*/*",
                "Content-Type": "application/json" 
                }
            headers2 = headers.generate().update({'accept-encoding': 'gzip, deflate, utf-8'})
            
            # rate limit of 100 requests per 5 minutes, 1 request per 3 seconds
            # We multiply the wait time by 2 since we also make a second request later on
            # The first request gets the list of pdf links and details of the research papers
            # The second request gets the actual pdf documetns
            if idx == 0:
                await asyncio.sleep(proc_idx_totalcount[0]*3.5)
            else:    
                await asyncio.sleep( max( (idx*3.5*conccurent_tasks)*1.1 - (time.time()-start_time), 0.0)  )
            
            # time.sleep(4)

            try:
                async with session.get(url, headers=headers1, timeout=60*conccurent_tasks ) as resp:
                    
                    if resp.status != 200:
                        break
                    else:
                        pass

                    resp_dict = await resp.content.read()
                    resp_dict = json.loads(resp_dict.decode())
                    
                    # break when no more pages left on website for query,
                    # semantic scholar api returns a total number of papers that match the query as ['total']
                    if resp_dict['total'] < (idx+1)*papers_per_query:
                        break
            except asyncio.TimeoutError as e:
                logging.warning(f"request timed out - {search_term} -\n\t{url}")
                break


            li_dict_papers = resp_dict['data']

            ## reformating author fields
            for idx_1 in range(len(li_dict_papers)):
                li_dict_papers[idx_1]['authors'] = ', '.join( ( dict_['name'] for dict_ in li_dict_papers[idx_1]['authors'] ))

            ## filtering for research papers that have at least min_citations citations
            for idx_2 in reversed(range(len(li_dict_papers))):
                dict_paper = li_dict_papers[idx_2]

                if dict_paper['citationCount'] < min_citations:
                    li_dict_papers.pop(idx_2)
                
            # extracting pdf documents        
            pdfs = []
            logger.info(f"\tDownloading {len(li_dict_papers)} for {search_term}")
            for idx_3, dict_ in enumerate(li_dict_papers):
                
                dict_ = li_dict_papers[idx_3]
                
                try:
                    # randomly select a float between 0.5 and 2.5 times the average time to download a pdf
                    stime = np.random.uniform( conccurent_tasks*0.75, conccurent_tasks*2.0 )

                    await asyncio.sleep( stime )

                    pdf = await (await session.get(dict_['openAccessPdf']['url'], cookies=resp.cookies,
                            headers=headers2, timeout=60*conccurent_tasks
                            )).content.read()

                except (ClientError):
                    pdf = "NAN"
                    # time.sleep(15.0)
                    logger.warning(f"ClientError - {search_term} -\n\t{url}")
                    await asyncio.sleep(conccurent_tasks*10)

                except(asyncio.TimeoutError):
                    pdf = "NAN"
                    logger.warning(f"TimeoutError - {search_term} -\n\t{url}")
                    await asyncio.sleep(conccurent_tasks*10)

                pdfs.append(pdf)
                if (idx_3+1)%(papers_per_query/5)==0:
                    logger.info(f"\tCurrently downloading pdf number {len(pdfs)} for {search_term}")


            # Filtering out invalid pdfs
            pdfs, titles, authors  = zip( *[ (pdf, d['title'], d['authors'] ) for d, pdf in zip(li_dict_papers,pdfs) if pdf[:4]==b'%PDF'  ]  )
            
            li_pdf.extend(pdfs)
            li_title.extend(titles)
            li_author.extend(authors)

            logger.info(f"\tDownloaded {len(li_pdf)} for {search_term}")
            
            # break pdf urls collected exceeds min citations
            if len(li_pdf)>=downloads_per_search_term:
                li_pdf = li_pdf[:downloads_per_search_term]
                li_title = li_title[:downloads_per_search_term]
                li_author = li_author[:downloads_per_search_term]
                logger.info(f"\tFinished Downloading for {search_term}")

                break


        outp = list( zip(li_pdf, li_title, li_author))
    return outp

def remove_duplicates(li_li_pdf_title_author: list[list[tuple[bytes|int,str,str]]]) -> list[list[tuple[bytes,str,str]]] :
    """Remove duplicate papers from the list of lists of papers.

    Args:
        li_li_pdf_title_author (list[list[tuple[int,str,str]]]): List of lists of papers.

    Returns:
        list[list[tuple[int,str,str]]]: List of lists of papers without duplicates.
    """
    
    unique_titles_author = set()
    
    for idx1 in range(len(li_li_pdf_title_author)):
        for idx2 in reversed(range(len(li_li_pdf_title_author[idx1]))):

            title = li_li_pdf_title_author[idx1][idx2][1]
            author = li_li_pdf_title_author[idx1][idx2][2]

            if (title,author) not in unique_titles_author:
                unique_titles_author.add( (title,author) )
            else:
                li_li_pdf_title_author[idx1].pop(idx2)
    
    return li_li_pdf_title_author

def save_pdfs(search_term, search_term_idx, li_pdf_title_author):
    
    # making directory
    dir_ = f'./data/researchpapers/pdf_format/{search_term_idx:02}'
    os.makedirs(dir_, exist_ok=True)

    with open( os.path.join(dir_,'search_term.txt'), 'w') as f:
        f.write(search_term)

    # Saving index mapping file_numbers to paper titles
    fp_index = os.path.join(dir_, 'index.csv')
    with open(fp_index, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(  [ (idx, title, author) for idx, (pdf, title, author) in enumerate(li_pdf_title_author) ] ) 

    # Saving documents
    for idx, (pdf, title, author) in enumerate(li_pdf_title_author):
        fp_file = os.path.join(dir_, f"{idx:03}.pdf")
        with open(fp_file, "wb") as f:
            f.write( pdf )
    
    return None

def load_pdfs(search_terms) -> list[ list[tuple[bytes,str,str]]  ]:
    """Load papers from saved pdf files."""

    # Loading the pdf, title and authors
    li_li_pdf_title_author = []

    for idx, search_term in enumerate(search_terms):
        dir_ = f'./data/researchpapers/pdf_format/{idx:02}'
        fp_index = os.path.join(dir_, 'index.csv')

        # Loading title and author
        with open(fp_index, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            li_title_author = [ (title, author) for idx, title, author in reader ]
        
        # Loading pdf docs
        li_pdf = []
        for file in sorted(os.listdir(dir_)):
            if file.endswith(".pdf"):
                with open(os.path.join(dir_, file), "rb") as f:
                    pdf = f.read()
                    li_pdf.append(pdf)

        # Combining pdf, title and author
        li_pdf_title_author = [ (pdf, title, author) for pdf, (title, author) in zip(li_pdf, li_title_author) ]

        li_li_pdf_title_author.append(li_pdf_title_author)
    
    return li_li_pdf_title_author

def extract_text_fitz(pdf:bytes) -> str:
    
    doc = fitz.Document( stream=pdf )
    text = ''
    for page in doc:
        text += page.get_text()

    return text

def extract_text_pdfminer(pdf:bytes) -> str:

    text = extract_text( BytesIO(pdf), caching=False )

    return text

def extract_text_pdftotext(pdf:bytes) -> str:

    pdf = pdftotext.PDF( BytesIO(pdf) )

    # Dropping pages after 'References' header, this usually includes appendix
    try:
        final_page = next((idx for idx, page in reversed(list(enumerate(pdf))) if 'References' in page))
    except StopIteration:
        final_page = len(pdf)+1

    pages_filtered = [page for idx, page in enumerate(pdf) if idx<final_page]

    l = len(pages_filtered)

    # end of page is indicated by \x0c
    # To discern whether text is continuing or new paragraph, we check if the first character is a capital letter in next page
    for page_idx, next_page_idx in zip( range(l), range(1,l) ):
        curr_page = pages_filtered[page_idx]
        next_page = pages_filtered[next_page_idx]

        # check next page is lower case and current page ends with \x0c and ends with end of line punctuation
        if next_page[0].islower() and (curr_page[-8:] == '\n\n\x0c') and (curr_page[-9] not in ['.', '!', '?']):
            curr_page[-8:] = ' '
            pages_filtered[page_idx] = curr_page
            
    txt = "".join( pages_filtered )

    return txt

def save_text(search_term:str, search_term_idx:int, li_txt_title_author: list[list[str]]):

    # making directory
    dir_ = f'./data/researchpapers/text_format/{search_term_idx:02}'
    os.makedirs(dir_, exist_ok=True)

    with open( os.path.join(dir_,'search_term.txt'), 'w') as f:
        f.write(search_term)

    # Saving index mapping file_numbers to paper titles
    fp_index = os.path.join(dir_, 'index.csv')
    with open(fp_index, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(  [ (idx, title, author) for idx, (txt, title, author) in enumerate(li_txt_title_author) ] ) 

    for idx, (txt, title, author) in enumerate(li_txt_title_author):
        # fp_file = os.path.join(dir_, f"{idx:03}.txt.gz")
        # with gz.open(fp_file, "wb") as f:
            # f.write( txt.encode('utf-8') )
        fp_file = os.path.join(dir_, f"{idx:03}.txt")
        with open(fp_file, "w") as f:
            f.write( txt )
              
def parse_args(parent_parser):
    if parent_parser != None:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
    else:
        parser = ArgumentParser()

    
    parser.add_argument('--downloads_per_search_term', default=100, type=int, help='Number of documents to download per search term')
    parser.add_argument('--min_citations', type=int, default=0, help='Minimum number of citations for a paper to have to be included in download')
    parser.add_argument('--mp_count', type=int, default=4, help='')
    parser.add_argument('--source', type=str, default='semantic_scholar', help='Which website to use for sourcing the research papers', choices=['google_scholar','semantic_scholar'])
    parser.add_argument('--pdf_parser', type=str, default='pdftotext', help='Which pdf parser to use', choices=['pdfminer','pdftotext','fitz'])
    parser.add_argument('--debugging', action='store_true', default=False, help='Whether to run in debuggging mode')

    parser.add_argument('--pdfs_downloaded', action='store_true',
                         default=False, help='Whether the pdfs for the documents have already been downloaded')
    
    args = parser.parse_known_args()[0]

    return args

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False, allow_abbrev = False)
    
    args = parse_args(parser)

    main(**vars(args))