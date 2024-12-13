import unittest
import numpy as np
import pandas as pd
from agent_based_modelling.calibration import get_b2i_network, get_i2i_network, calibrate, interpretable_entropy 
import itertools
from unittest.mock import patch
import random
class TestCalibration(unittest.TestCase):
    
    def setUp(self):
        self.model_size = '7bn'
        self.budget_item_count = 25
        self.indic_count = 100
        self.start_year = 2013
        self.end_year = 2017
        self.T = int(5*(self.end_year - self.start_year))

        self.threshold = 0.1
        self.low_precision_counts = 5
        self.increment = 10
        self.parallel_processes = 1
        self.verbose = True
        self.time_experiments = True

        
        # Dummy data
        self.indic_start = np.random.rand(self.indic_count)/5 # shape: (indic_count)
        self.indic_final = self.indic_start + np.random.rand(self.indic_count)/5

        self.success_rates = np.random.rand(self.indic_count) # shape: (indic_count)
        self.R = np.ones(self.indic_count ) # shape: (indic_count)
        self.qm = np.random.rand(self.indic_count) # shape: (indic_count)
        self.rl = np.random.rand(self.indic_count ) # shape: (indic_count)
        self.Bs = np.random.rand( self.budget_item_count, self.T ) # shape: (budget_item_count, years)
        
        
        self.i2i_network = np.random.rand(self.indic_count, self.indic_count)
        # Create an neverending iterable of integers between 0 and self.budget_item_count
        bi_idx_iter = itertools.cycle( list( range(self.budget_item_count) ))
        # B_dict creates a mapping from indicators (keys) to a list of budget items (values)
        # all budget items must be included in the mapping
        self.B_dict = { k: sorted( list(itertools.islice(bi_idx_iter, 2)))  
            for k in range(self.indic_count) }
        
        
        indicator1_names = [f"ind1_{i+1}" for i in range(20)]
        indicator2_names = [f"ind1_{i+2}" for i in range(20)]
        weights = np.random.uniform(0, 5, 20)
        
        self.i2i_networks = {}
        self.i2i_networks['CPUQ_multinomial'] = pd.DataFrame(
            {'indicator1': indicator1_names,
            'indicator2': indicator2_names,
            'weight': [ {'mean': v, 'scaled_mean':v/5} for v in weights ] ,
            'distribution': 
            [ 
                [{0:0.0, 1:0.0, 2:1.0, 3:0.0, 4:0.0, 5:0.0}] for _ in range(5) # entropy = 1.0
             ] + [
                [{0:0.0, 1:0.3, 2:0.6, 3:0.1, 4:0.0, 5:0.0}] for _ in range(5) # entropy =  0.4988
             ] + [
                [{0:0.1, 1:0.18, 2:0.18, 3:0.18, 4:0.18, 5:0.18}] for _ in range(10) # entropy = 0.0101
             ]
            }
            )
        
        self.i2i_networks['verbalize'] = pd.DataFrame(
            {'indicator1': indicator1_names,
            'indicator2': indicator2_names,
            'preds': [ [{'mean': w}] for w in weights ] ,
            'weight': [ {'mean':w ,'scaled_mean':w/5} for w in weights] }
        )
        self.i2i_networks['entropy'] = pd.DataFrame({
                                    "indicator1": indicator1_names,
                                    "indicator2": indicator2_names,
                                    "weight":  [ {'mean': w} for w in weights ] })
        self.i2i_networks['CPUQ_multinomial_adj'] = self.i2i_networks['CPUQ_multinomial']

        self.i2i_networks['ccdr'] = pd.DataFrame({
            'From': list(range(20)),
            'To': [i+1 for i in range(20)],
            'Weight': list(weights/5)
        })


    def test_get_b2i_network(self):
        for b2i_method in ['ea', 'verbalize', 'CPUQ_binomial']:
            with self.subTest(b2i_method=b2i_method):
                B_dict = get_b2i_network(b2i_method, self.model_size)
                self.assertIsInstance(B_dict, dict)
                self.assertEqual(len(B_dict), 415)

    @patch('pandas.read_csv')               
    def test_get_i2i_network(self, mock_read_csv):
        for i2i_method, i2i_threshold in [ ('CPUQ_multinomial',0.2), 
                                              ('CPUQ_multinomial',0.7),
                                              ('ccdr',0.1),
                                              ('zero',None), ('verbalize',2),
                                              ('entropy',None) ]:
            
            with self.subTest(i2i_method=i2i_method, i2i_threshold=i2i_threshold):
                
                if i2i_method =='zero':
                    pass
                elif i2i_method == 'ccdr':
                    mock_read_csv.side_effect = [self.i2i_networks[i2i_method]]
                else:
                    mock_read_csv.side_effect = [
                        self.i2i_networks[i2i_method],
                        pd.DataFrame({'indicator_name': list(set( 
                            self.i2i_networks[i2i_method].indicator1.tolist() +
                            self.i2i_networks[i2i_method].indicator2.tolist())) })
                    ]

                i2i_network = get_i2i_network(i2i_method, self.indic_count, 
                                              self.model_size, i2i_threshold)



                self.assertIsInstance(i2i_network, np.ndarray)
                self.assertEqual(i2i_network.shape, (self.indic_count, self.indic_count))
                
                if i2i_method in ['verbalize', 'entropy', 'CPUQ_multinomial', 'ccdr' ]:
                    self.assertTrue(np.any(i2i_network != 0))
                
                if i2i_threshold is not None:
                    if i2i_method == 'ccdr':
                        filtered_count = sum([1 for w in self.i2i_networks[i2i_method]['Weight'].tolist() if w>= i2i_threshold])
                        self.assertEqual(np.count_nonzero(i2i_network), filtered_count)
                    
                    elif i2i_method == 'verbalization':
                        filtered_count = sum([1 for w in self.i2i_networks[i2i_method]['weight'].tolist() if w>= i2i_threshold])
                        self.assertEqual(np.count_nonzero(i2i_network), filtered_count)
                    
                    elif i2i_method in ['CPUQ_multinomial', 'CPUQ_multinomial_adj']:
    
                        distributions = self.i2i_networks[i2i_method]['distribution'].values.tolist()
                        entropies = [interpretable_entropy(d[0]) for d in distributions]
                        filtered_count = sum([1 for e in entropies if e>= i2i_threshold])  
                        self.assertTrue(np.all( np.array(entropies) <= 1))       
                    
                        self.assertEqual(np.count_nonzero(i2i_network), filtered_count)

    def test_calibrate(self):
        for b2i_method, i2i_method, parrallel_processes, time_experiments in [
            # ('ea', 'zero', None, False),
            #  ('ea','verbalize', 2, True),
        #  ('ea', 'entropy', None, False),
        #   ('ea','CPUQ_multinomial', None, False),
         ('verbalize','verbalize', 4, False), 
        #  ('CPUQ_binomial','CPUQ_multinomial', None, False),
        #  ('ea','ccdr', None, False)
         ]:
            
            with self.subTest(b2i_method=b2i_method, i2i_method=i2i_method, 
                parrallel_processes=parrallel_processes, time_experiments=time_experiments):
                                
                df_output = calibrate(
                    indic_start = self.indic_start,
                    indic_final = self.indic_final,

                    success_rates=self.success_rates,
                    R=self.R,
                    qm=self.qm,
                    rl=self.rl,
                    Bs=self.Bs,
                    B_dict=self.B_dict,
                    T=self.T,
                    i2i_network=self.i2i_network,

                    parallel_processes=parrallel_processes,
                    threshold = self.threshold,
                    low_precision_counts=self.low_precision_counts,
                    increment=self.increment,
                    verbose=self.verbose,
                    time_experiments=self.time_experiments,

                    mc_simulations=1
                )

                self.assertIsInstance(df_output, dict)
                
                self.assertIn('parameters', df_output)
                self.assertEqual( df_output['parameters'].shape[1], 8)
                
                if time_experiments:
                    self.assertIn('time_elapsed', df_output)
                    self.assertIsInstance(df_output['time_elapsed'], float)


                    self.assertIn('iterations', df_output)
                    self.assertIsInstance(df_output['iterations'], int)

    
      
if __name__ == '__main__':
    unittest.main()
