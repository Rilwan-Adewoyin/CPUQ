import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

from agent_based_modelling.imputation import load_currI_fBs_frl_fG, impute_indicators, load_true_indicators

class TestImputation(unittest.TestCase):

    def setUp(self):
        # Create mock data if needed
        
        self.calibration_start_year = 2016
        self.impute_start_year = 2019
        calibration_end_year = self.impute_start_year -1
        calibration_years =  calibration_end_year - self.calibration_start_year + 1
        self.impute_years = 2
        self.tft = time_refinement_factor = 2

        self.mock_data = {
            'indicators': pd.DataFrame({
                '2016': [0.5, 0.7],
                '2017': [0.2, 0.4],
                '2018': [0.5, 0.7],
                '2019': [0.6, 0.75],
                '2020': [0.65, 0.8],
                'indicator_name': ['indicator_0', 'indicator_1'],
                'rl': [0.8, 0.7],
                'R': [1.0, 1.0],
                'qm': [0.88, 0.44],
            }),
            'expenditure': pd.DataFrame({
                'time_refinement_factor': [self.tft,self.tft],
                'seriesName':['expenditure_0', 'expenditure_1'],
                '0': [0.5, 0.7],
                '1': [0.2, 0.4],
                '2': [0.5, 0.7],
                '3': [0.6, 0.75],
                '4': [0.65, 0.8],
                '5': [0.65, 0.8],
                '6': [0.65, 0.8],
                '7': [0.65, 0.85],
                '8': [0.75, 0.85],
                '9': [0.75, 0.85],

            })
        }
        self.indic_count = self.mock_data['indicators'].shape[0]

        self.i2i_network = np.random.rand(self.indic_count, self.indic_count)
        self.b2i_network = { 0: [1], 1:[0,1] }
        
        # calibrated_params = pd.DataFrame({
        #             'alphas': [0.5, 0.7],
        #             'alphas_prime': [0.5, 0.7],
        #             'betas': [0.2, 0.4],

        # })
        calibrated_hparams = {
                    'calibration_start_year': self.calibration_start_year,
                    'calibration_end_year': calibration_end_year
        }
        # self.mock_data['calibrated_params'] = calibrated_params
        self.mock_data['calibrated_hparams'] = calibrated_hparams

        
        self.mock_data['model_params'] =  pd.DataFrame(
                columns=['alpha', 'alpha_prime', 'beta', 'T', 'error_alpha', 'error_beta', 'GoF_alpha', 'GoF_beta'],
                data=[
                    [4.712208662238478e-06, 4.190268510921154e-05, 1.3903237720967019, float('nan'), 1.4278519667176859e-05, 0.003491841715176358, 0.9844982821828393, 0.9919816968021876],
                    [0.000168227856490462, 2.3487589062989112e-06, 0.9808585708091208, float('nan'), 6.230640507132179e-05, 0.002951832136905308, 0.9791373567714184, 0.9932217187967359],
            
                ]
            )
        

    @patch('yaml.safe_load')
    @patch('glob.glob')   # Patching glob.glob
    @patch('pandas.read_csv')
    def test_load_currI_fBs_frl_fG(self, mock_read_csv, mock_glob, mock_safe_load):
        mock_glob.return_value = [ 'params_v01.csv']
        mock_read_csv.side_effect = [ self.mock_data['indicators'], self.mock_data['expenditure']]
        mock_safe_load.return_value = self.mock_data['calibrated_hparams']
        
        # Calling the function with mock data
        current_I, fBs, frl, fG, refinement_factor = load_currI_fBs_frl_fG( self.impute_start_year, self.impute_years)
        
        tft = self.mock_data['expenditure']['time_refinement_factor'][0]
        
        # pre assertion checks
        idx_start_year = list(self.mock_data['indicators'].columns).index(str(self.impute_start_year - 1))
        idx_end_year = idx_start_year + self.impute_years
        impute_start_year_idx = self.impute_start_year - self.mock_data['calibrated_hparams']['calibration_start_year']
        
        np.testing.assert_array_equal(current_I, self.mock_data['indicators'][str(self.impute_start_year-1)].values )
        np.testing.assert_array_equal(fBs, self.mock_data['expenditure'][ list( str(idx) for idx in  range(impute_start_year_idx*tft, (idx_end_year+1)*tft)) ].values )
        np.testing.assert_array_equal(frl, self.mock_data['indicators']['rl'].values )
        np.testing.assert_array_equal(fG, self.mock_data['indicators'][str(self.impute_start_year+self.impute_years-1)].values )

    @patch('pandas.read_csv')
    @patch('agent_based_modelling.imputation.load_model_kwargs')
    @patch('agent_based_modelling.ppi.run_ppi')
    def test_impute_indicators(self, mock_run_ppi, mock_load_model_kwargs, mock_read_csv):
        mock_run_ppi.return_value = (np.array([[0.6, 0.75]]), None, None, None, None, None)
        mock_load_model_kwargs.return_value = self.mock_data['model_params']
        mock_read_csv.return_value = self.mock_data['indicators']

        idx_start_year = list(self.mock_data['indicators'].columns).index(str(self.impute_start_year - 1))
        idx_end_year = idx_start_year + self.impute_years
        idx_end_year = idx_start_year + self.impute_years
        impute_start_year_idx = self.impute_start_year - self.mock_data['calibrated_hparams']['calibration_start_year']
        tft = self.mock_data['expenditure']['time_refinement_factor'][0]
        
        fBs = self.mock_data['expenditure'][ list( str(idx) for idx in  range(impute_start_year_idx*tft, (idx_end_year+1)*tft)) ].values
        frl = self.mock_data['indicators']['rl'].values 
        fG = self.mock_data['indicators'][str(self.impute_start_year+self.impute_years-1)].values 

        I0 = self.mock_data['indicators'][str(self.impute_start_year-1)].values

        # Calling the function with mock data
        model_params = mock_load_model_kwargs.return_value = self.mock_data['model_params']
        
        results = impute_indicators(self.impute_years, self.tft,
            I0, fBs, frl, fG, self.i2i_network, self.b2i_network,
            model_params, parallel_processes=1 )

        self.assertIsNotNone(results)  # Add more assertions based on expected results
        

    @patch('pandas.read_csv')
    def test_load_true_indicators(self, mock_read_csv):
        mock_read_csv.return_value = self.mock_data['indicators']

        # Calling the function with mock data
        indicator_values, indicator_names = load_true_indicators(2019, 2)
        
        np.testing.assert_array_equal(indicator_names, self.mock_data['indicators']['indicator_name'].values )
        cols = [str(year) for year in range(self.impute_start_year, self.impute_start_year+self.impute_years) ]
        np.testing.assert_array_equal(indicator_values, self.mock_data['indicators'][cols].values )
    
    def tearDown(self):
        # Clean up if needed
        pass

if __name__ == '__main__':
    unittest.main()
