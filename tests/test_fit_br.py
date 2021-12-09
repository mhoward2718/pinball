"""Tests for `quantreg` package.

Borrow some tests from
https://github.com/statsmodels/statsmodels/blob/6b4aa33563ab639d168525dde0ef86c8e5c83d68/statsmodels/regression/tests/test_quantile_regression.py
"""
from quantreg.br import fit_br
import pytest
import numpy as np
from unittest import TestCase
from unittest.mock import patch
from unittest.mock import Mock

class TestGetQN(TestCase):
    
    # Mock wls weights.
    # Mock WLS?
    def test_iid(self):
        X = np.array([[1,2,3],[4,5,6]])
        y = np.array([10,100,100])
        tau = 0.5
        result = fit_br.get_qn(X, y, tau, iid=True)
    
    def test_not_iid(self):
        X = np.array([[1,2,3],[4,5,6]])
        y = np.array([10,100,100])
        tau = 0.5
        
        fit_result = Mock(resid=np.array([19.5,19.6,19.7]))
        # How do I mock WLS, then have fit() return fit_result?
        with patch('statsmodels.regression.linear_model.WLS'):
            result = fit_br.get_qn(X, y, tau, iid=False)
    
    def test_r_cases(self):
        pass
        
class TestFitBR(TestCase):
    # Will need to mock rqbr
    def test_ci(self):
        pass
    
    def test_single_predictor(self):
        pass
    
    def test_singular_input(self):
        pass
    
    def test_single_quantile(self):
        pass
    
    def test_all_quantile(self):
        pass
    
    # TODO: Define tests that actually run fortran and verify result

class TestGetWLSWeights(TestCase):
    """Test that WLS weights are correctly estimated"""
    
    # @patch('quantreg.br.fit_br.fit_br')
    def test_larger_than_eps_equals_weights(self):
        X = np.array([[1,2,3],[4,5,6]])
        y = np.array([10,100,100])
        tau = 0.5
        # Choose return values that result in 2*h blah blah getting returned
        blo = Mock(coef=np.array([1.7,1.8,1.9]))
        bhi = Mock(coef=np.array([19.5,19.6,19.7]))
        
        with patch('quantreg.br.fit_br.fit_br', side_effect = [bhi, blo]):
            pass
    
    def test_smaller_than_eps_equals_eps(self):
        # Choose return values that result in eps being returned
        pass
    
    # What is 'percent fis' anyway?
    def test_dyhat_print_warning(self):
        X = np.array([[1,2,3],[4,5,6]])
        y = np.array([10,100,100])
        tau = 0.5
        # Choose return values that result in 2*h blah blah getting returned
        blo = np.array([0.7,0.8,0.9])
        bhi = np.array([0.5,0.6,0.7])
        # Possibly patch bandwidth also
        # Not really a unittest otherwise
        blo = Mock(coef=np.array([0.7,0.8,0.9]))
        bhi = Mock(coef=np.array([0.5,0.6,0.7]))
        
        with patch('quantreg.br.fit_br.fit_br', side_effect = [bhi, blo]):
            with patch('builtins.print'):
                result = fit_br.get_wls_weights(X, y, tau)
                # Verify result warning was called
                print.assert_called_with("Percent fis <= 0: 20.0")

    # What is 'percent fis' anyway?
    def test_dyhat_no_print_warning(self):
        X = np.array([[1,2,3],[4,5,6]])
        y = np.array([10,100,100])
        tau = 0.5
        # Choose return values that result in 2*h blah blah getting returned
        blo = Mock(coef=np.array([1.7,1.8,1.9]))
        bhi = Mock(coef=np.array([19.5,19.6,19.7]))
        # Maybe not worth thinking too hard how to test this
        with patch('quantreg.br.fit_br.fit_br', side_effect = [bhi, blo]):
            with patch('builtins.print'):
                result = fit_br.get_wls_weights(X, y, tau)
                # Verify result warning was called
                self.assertFalse(print.called)
                          
    def test_compare_with_r_values(self):
        """Run a few test inputs and ensure the result is equal to 
        what the R quantreg package gives
        
        Good idea to define a few test sets in a 'test_data.py' file or similar
        
        """
        pass
    
class TestDeriveBRParams(TestCase):
    """Test that solver parameters are derived correctly
    """
    # TODO: Probably should implement checks on the types, shapes, etc of input arguments
    # then create unit tests for those checks
    
    # TODO: 3x3 is probably too trivial
    # Try testing a case where number of columns and rows are different
    def test_single_quantile(self):
        X = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 3x3 array
        y = np.array([[50.,70.,90.]])
        tau = 0.90
        actual_params = fit_br.derive_br_params(X, y, tau)
        expected_params = fit_br.BRParams(m=3,
                nn=np.int32(3),
                m5=np.int32(3 + 5), 
                n3=np.int32(3 + 3),
                n4=np.int32(3 + 4),
                a=X,
                b=y,
                t=0.90,
                toler=np.finfo(np.float64).eps ** (2/3),
                ift=np.int32(1), 
                x=np.zeros(3, np.float64),
                e=np.zeros(3, np.float64),
                s=np.zeros(3,dtype=np.int32),
                wa=np.zeros(((3 + 5),(3 + 4)), dtype=np.float64),
                wb=np.zeros(3, dtype=np.float32),
                nsol=np.int32(2),
                ndsol=np.int32(2),
                sol=np.zeros(((3 + 3), 2), dtype=np.float64),
                dsol=np.zeros((3, 2), dtype=np.float64),
                lsol=np.int32(0),
                h=np.zeros((3,2), dtype=np.int32),
                qn=np.zeros(3, dtype=np.float64),
                cutoff=np.float64(0), 
                ci=np.zeros((4,3), dtype=np.float64),
                tnmat=np.zeros((4,3), dtype=np.float64),
                big=np.finfo(np.float64).max,
                lci1=np.bool_(False))
        
        # It looks like because these tuples contain arrays, we can't assert equality of 
        # the whole tuple in a single comparison
        
        self.assertTrue(actual_params.m == expected_params.m)
        self.assertTrue(actual_params.nn == expected_params.nn)
        self.assertTrue(actual_params.m5 == expected_params.m5)
        self.assertTrue(actual_params.n3 == expected_params.n3)
        self.assertTrue(actual_params.n4 == expected_params.n4)
        np.testing.assert_array_equal(actual_params.a, expected_params.a)
        np.testing.assert_array_equal(actual_params.b, expected_params.b)
        self.assertTrue(actual_params.t == expected_params.t)
        self.assertTrue(actual_params.toler == expected_params.toler)
        self.assertTrue(actual_params.ift == expected_params.ift)
        np.testing.assert_array_equal(actual_params.x, expected_params.x)
        np.testing.assert_array_equal(actual_params.e, expected_params.e)
        np.testing.assert_array_equal(actual_params.s, expected_params.s)
        np.testing.assert_array_equal(actual_params.wa, expected_params.wa)
        np.testing.assert_array_equal(actual_params.wb, expected_params.wb)
        self.assertTrue(actual_params.nsol == expected_params.nsol)
        self.assertTrue(actual_params.ndsol == expected_params.ndsol)
        np.testing.assert_array_equal(actual_params.sol, expected_params.sol)
        np.testing.assert_array_equal(actual_params.dsol, expected_params.dsol)
        self.assertTrue(actual_params.lsol == expected_params.lsol)
        np.testing.assert_array_equal(actual_params.h, expected_params.h)
        np.testing.assert_array_equal(actual_params.qn, expected_params.qn)
        self.assertTrue(actual_params.cutoff == expected_params.cutoff)
        np.testing.assert_array_equal(actual_params.ci, expected_params.ci)
        np.testing.assert_array_equal(actual_params.tnmat, expected_params.tnmat)        
        self.assertTrue(actual_params.big == expected_params.big)
        self.assertTrue(actual_params.lci1 == expected_params.lci1)
        
    def test_all_quantiles(self):
        X = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 3x3 array
        y = np.array([[50.,70.,90.]])
        tau = None
        actual_params = fit_br.derive_br_params(X, y, tau)
        expected_params = fit_br.BRParams(m=3,
                nn=np.int32(3),
                m5=np.int32(3 + 5), 
                n3=np.int32(3 + 3),
                n4=np.int32(3 + 4),
                a=X,
                b=y,
                t=-1,
                toler=np.finfo(np.float64).eps ** (2/3),
                ift=np.int32(1), 
                x=np.zeros(3, np.float64),
                e=np.zeros(3, np.float64),
                s=np.zeros(3,dtype=np.int32),
                wa=np.zeros(((3 + 5),(3 + 4)), dtype=np.float64),
                wb=np.zeros(3, dtype=np.float32),
                nsol=np.int32(9),
                ndsol=np.int32(9),
                sol=np.zeros(((3 + 3), 9), dtype=np.float64),
                dsol=np.zeros((3, 9), dtype=np.float64),
                lsol=np.int32(0),
                h=np.zeros((3,9), dtype=np.int32),
                qn=np.zeros(3, dtype=np.float64),
                cutoff=np.float64(0), 
                ci=np.zeros((4,3), dtype=np.float64),
                tnmat=np.zeros((4,3), dtype=np.float64),
                big=np.finfo(np.float64).max,
                lci1=np.bool_(False))
        
        self.assertTrue(actual_params.m == expected_params.m)
        self.assertTrue(actual_params.nn == expected_params.nn)
        self.assertTrue(actual_params.m5 == expected_params.m5)
        self.assertTrue(actual_params.n3 == expected_params.n3)
        self.assertTrue(actual_params.n4 == expected_params.n4)
        np.testing.assert_array_equal(actual_params.a, expected_params.a)
        np.testing.assert_array_equal(actual_params.b, expected_params.b)
        self.assertTrue(actual_params.t == expected_params.t)
        self.assertTrue(actual_params.toler == expected_params.toler)
        self.assertTrue(actual_params.ift == expected_params.ift)
        np.testing.assert_array_equal(actual_params.x, expected_params.x)
        np.testing.assert_array_equal(actual_params.e, expected_params.e)
        np.testing.assert_array_equal(actual_params.s, expected_params.s)
        np.testing.assert_array_equal(actual_params.wa, expected_params.wa)
        np.testing.assert_array_equal(actual_params.wb, expected_params.wb)
        self.assertTrue(actual_params.nsol == expected_params.nsol)
        self.assertTrue(actual_params.ndsol == expected_params.ndsol)
        np.testing.assert_array_equal(actual_params.sol, expected_params.sol)
        np.testing.assert_array_equal(actual_params.dsol, expected_params.dsol)
        self.assertTrue(actual_params.lsol == expected_params.lsol)
        np.testing.assert_array_equal(actual_params.h, expected_params.h)
        np.testing.assert_array_equal(actual_params.qn, expected_params.qn)
        self.assertTrue(actual_params.cutoff == expected_params.cutoff)
        np.testing.assert_array_equal(actual_params.ci, expected_params.ci)
        np.testing.assert_array_equal(actual_params.tnmat, expected_params.tnmat)        
        self.assertTrue(actual_params.big == expected_params.big)
        self.assertTrue(actual_params.lci1 == expected_params.lci1)

