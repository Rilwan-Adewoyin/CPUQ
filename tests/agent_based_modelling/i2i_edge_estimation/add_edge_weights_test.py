import unittest
from agent_based_modelling.i2i_edge_estimation.add_i2i_edge_weights import entropy_edge_weights
from scipy.stats import entropy

class TestEntropyEdgeWeights(unittest.TestCase):

    def test_entropy_edge_weights(self):
        
        li_records = [
            {'pred_aggregated': {'Yes': 0.9, 'No': 0.1}},
            {'pred_aggregated': {'Yes': 0.5, 'No': 0.5}},
            {'pred_aggregated': {'Yes': 0.2, 'No': 0.8}},
        ]

        
        results = entropy_edge_weights(li_records)

        # Check for number of records
        self.assertEqual(len(results), 3)
        
        # Check for first record
        self.assertEqual(results[0]['mean'], 1 - entropy([0.9, 0.1], base=2))

        # Check for second record
        self.assertEqual(results[1]['mean'], 1 - entropy([0.5, 0.5], base=2))

        # Check for third record
        self.assertEqual(results[2]['mean'], 1 - entropy([0.2, 0.8], base=2))

if __name__ == "__main__":
    unittest.main()
