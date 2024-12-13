import pytest
import os
import sys
sys.path.append(os.getcwd())

import requests
from data.finetune.scrape_research_papers import extract_text_pdftotext, remove_duplicates


@pytest.fixture
def pdf_data():
    """Sample pdf data to test extract_text_pdftotext function"""
    url = """https://www.health.org.uk/sites/default/files/2020-03/Health%20Equity%20in%20England_The%20Marmot%20Review%2010%20Years%20On_executive%20summary_web.pdf"""
    
    response = requests.get(url)
    return response.content

@pytest.fixture
def li_li_pdf_title_author():
    """Sample data to test remove_duplicates function"""
    sample_data = [
        [(1, "Title 1", "Author 1"), (2, "Title 2", "Author 2"), (3, "Title 3", "Author 3")],
        [(4, "Title 1", "Author 1"), (5, "Title 2", "Author 2"), (6, "Title 4", "Author 4")],
    ]

    return sample_data

@pytest.fixture
def li_li_pdf_title_author_duplicates_removed():
    """Expected data after removing duplicates"""
    expected_data = [
        [(1, "Title 1", "Author 1"), (2, "Title 2", "Author 2"), (3, "Title 3", "Author 3")],
        [(6, "Title 4", "Author 4")],
    ]

    return expected_data

def test_extract_text_pdftotext(pdf_data):
    """Test the extract_text_pdftotext function"""

    extracted_text = extract_text_pdftotext(pdf_data)

    # Check if the extracted text is not empty
    assert len(extracted_text)>0, "Extracted text should not be empty"

    print("All tests passed.")

def test_remove_duplicates(li_li_pdf_title_author: list[list[tuple[bytes|int,str,str]]], li_li_pdf_title_author_duplicates_removed: list[list[tuple[int,str,str]]]):
    """Test the remove_duplicates function"""
    
    actual_data = remove_duplicates(li_li_pdf_title_author)

    # Check if the extracted text is not empty
    assert actual_data == li_li_pdf_title_author_duplicates_removed, "Actual data should be equal to expected data"

    print("All tests passed.")