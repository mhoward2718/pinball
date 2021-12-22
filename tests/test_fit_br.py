"""Tests for `pinball` package.

TODO: Borrow some tests from
https://github.com/statsmodels/statsmodels/blob/6b4aa33563ab639d168525dde0ef86c8e5c83d68/statsmodels/regression/tests/test_quantile_regression.py
"""
from pinball.br import fit_br
import pytest
import numpy as np
from unittest import TestCase
from unittest.mock import patch
from unittest.mock import Mock

class TestGetRankInversionIntervals(TestCase):

    def test_iid(self):
        X = np.array([[1,2,3],
                      [4,5,6],
                      [10,20,30],
                      [1,10,100],
                      [7,5,3]])
        y = np.array([10,100,1000,5,25])
        tau = 0.5
        bw = Mock(return_value=1.0)
        actual_qn = fit_br.get_rank_inversion_intervals(X, y, tau, iid=True, bandwidth=bw)
        expected_qn = np.array([ 18.28325628, 37.52621247, 2878.87865655])
        np.testing.assert_array_almost_equal(actual_qn, expected_qn)

    def test_not_iid(self):
        X = np.array([[1,2,3],
                      [4,5,6],
                      [10,20,30],
                      [1,10,100],
                      [7,5,3]])
        y = np.array([10,100,1000,5,25])
        tau = 0.5
        bw = Mock(return_value=1.0)
        fit_result1 = Mock(resid=np.array([5.0, 10.0]))
        fit_result2 = Mock(resid=np.array([2.0, 4.0]))
        fit_result3 = Mock(resid=np.array([3.0, 7.0]))
        mock_model = Mock()
        mock_wls = Mock(return_value=mock_model)
        mock_model.fit.side_effect = [fit_result1, fit_result2, fit_result3]
        # Will be called three times
        #
        wls_weights = np.array([1.0,1.5,2.0])
        # This doesn't really test that each call to wls uses all but one of the
        # columns though
        with patch('pinball.br.fit_br.get_wls_weights', return_value=wls_weights):
            result = fit_br.get_rank_inversion_intervals(X, y, tau, iid=False, bandwidth=bw, wls=mock_wls)

        # Each column in result is the sum of the squares of the other columns
        # in fit_results
        expected_result = np.array([25.0+100.0,
                                    4.0+16.0,
                                    9.0+49.0])
        self.assertEqual(X.shape[1], result.shape[0])
        np.testing.assert_array_equal(result, expected_result)
        self.assertEqual(mock_wls.call_count, 3)
        calls = mock_wls.call_args_list
        expected_calls = [[np.array([[2,3],[5,6],[20,30],[10,100],[5,3]]), np.array([1,4,10,1,7]), wls_weights],
                          [np.array([[1,3],[4,6],[10,30],[1,100],[7,3]]), np.array([2,5,20,10,5]), wls_weights],
                          [np.array([[1,2],[4,5],[10,20],[1,10],[7,5]]), np.array([3,6,30,100,3]), wls_weights]
                         ]
        for (actual, expected) in zip(calls, expected_calls):
            # Note that positional and keyword args are stored as elements of a
            # tuple. Positional args are call[0], which is a list. The order
            # of the list is the order of the arguments.
            # keyword args are call[1], which is a dict of kwarg: value.
            np.testing.assert_array_equal(actual[0][0], expected[0])
            np.testing.assert_array_equal(actual[0][1], expected[1])
            # Keyword arg
            np.testing.assert_array_equal(actual[1]['weights'], expected[2])

    def test_r_cases(self):
        pass

class TestFitBR(TestCase):
    """Will need to mock rqbr"""

    def test_matrix_condition_error(self):
        X = np.array([[1,1,3],
              [4,4,6],
              [10,10,30],
              [1,1,100],
              [7,7,3]], dtype=np.float64)
        y = np.array([10,100,1000,5,25])
        tau = 0.5

        with pytest.raises(Exception) as execinfo:
            fit_br.fit_br(X, y, tau)
        self.assertEqual(execinfo.value.args[0], "Singular design matrix")

    """
    This is easy but I want a cleaner way to mock multiple classes and functions
    Opening each within a context(?) manager gets ugly because of the indents and
    repeated scopes

    """
    @patch('pinball.br.fit_br.derive_br_params')
    @patch('pinball_native.rqbr')
    @patch('pinball.br.fit_br.get_qn')
    def test_ci(self, mock_derive_params, mock_rqbr, mock_get_qn):
        X = np.array([[1,2,3],
                      [4,5,6],
                      [10,20,30],
                      [1,10,100],
                      [7,5,3]], dtype=np.float64)
        y = np.array([10,100,1000,5,25])
        tau = 0.5
        params = Mock()
        params.nn = 5
        mock_rqbr()
        mock_derive_params.return_value = params
        fit_br.fit_br(X, y, tau, ci = True)

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

    def test_larger_than_eps_equals_weights(self):
        X = np.array([[1,2,3], [4,5,6]])
        y = np.array([10,100,100])
        tau = 0.5
        blo = Mock(coef=np.array([1.7, 1.8, 1.9]))
        bhi = Mock(coef=np.array([19.5, 19.6, 19.7]))
        bw = Mock(return_value=1)

        with patch('pinball.br.fit_br.fit_br', side_effect = [bhi, blo]):
            actual = fit_br.get_wls_weights(X, y, tau, bandwidth=bw)

        expected = np.array([0.01872659, 0.00749064])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_smaller_than_eps_equals_eps(self):
        X = np.array([[1,2,3], [4,5,6]])
        y = np.array([10,100,100])
        tau = 0.5
        blo = Mock(coef=np.array([1.7, 1.8, 1.9]))
        bhi = Mock(coef=np.array([19.5, 19.6, 19.7]))
        bw = Mock(return_value=.0000000001)

        with patch('pinball.br.fit_br.fit_br', side_effect = [bhi, blo]):
            actual = fit_br.get_wls_weights(X, y, tau, bandwidth=bw)
        eps = np.finfo(np.float64).eps ** (2/3)
        expected = np.array([eps, eps])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_dyhat_print_warning(self):
        X = np.array([[1,2,3],[4,5,6]])
        y = np.array([10,100,100])
        tau = 0.5
        blo = np.array([0.7,0.8,0.9])
        bhi = np.array([0.5,0.6,0.7])
        blo = Mock(coef=np.array([0.7,0.6,0.7]))
        bhi = Mock(coef=np.array([0.5,0.6,0.8]))
        bw = Mock(return_value=1)
        with patch('pinball.br.fit_br.fit_br', side_effect = [bhi, blo]):
            with patch('builtins.print'):
                result = fit_br.get_wls_weights(X, y, tau, bandwidth=bw)
                # Verify result warning was called
                print.assert_called_with("Percent fis <= 0: 50.0")

    # What is the meaning of FIS? Is it referencing some f_i in a paper?
    def test_dyhat_no_print_warning(self):
        X = np.array([[1,2,3],[4,5,6]])
        y = np.array([10,100,100])
        tau = 0.5
        blo = Mock(coef=np.array([1.7,1.8,1.9]))
        bhi = Mock(coef=np.array([19.5,19.6,19.7]))
        bw = Mock(return_value=1)
        with patch('pinball.br.fit_br.fit_br', side_effect = [bhi, blo]):
            with patch('builtins.print'):
                result = fit_br.get_wls_weights(X, y, tau, bandwidth=bw)
                # Verify result warning was called
                self.assertFalse(print.called)

    def test_compare_with_r_values(self):
        """Run a few test inputs and ensure the result is equal to
        what the R pinball package gives

        Good idea to define a few test sets in a 'test_data.py' file or similar

        """
        pass

class TestDeriveBRParams(TestCase):
    """Test that solver parameters are derived correctly
    """
    # TODO: Probably should implement checks on the types, shapes, etc of input arguments
    # then create unit tests for those checks

    def test_single_quantile(self):
        X = np.array([[1,2,3],
                      [4,5,6],
                      [10,20,30],
                      [1,10,100],
                      [7,5,3]])
        y = np.array([10,100,1000,5,25])
        tau = 0.90
        actual_params = fit_br.derive_br_params(X, y, tau)
        expected_params = fit_br.BRParams(m=5,
                nn=np.int32(3),
                m5=np.int32(5 + 5),
                n3=np.int32(3 + 3),
                n4=np.int32(3 + 4),
                a=X,
                b=y,
                t=0.90,
                toler=np.finfo(np.float64).eps ** (2/3),
                ift=np.int32(1),
                x=np.zeros(3, np.float64),
                e=np.zeros(5, np.float64),
                s=np.zeros(5,dtype=np.int32),
                wa=np.zeros(((5 + 5),(3 + 4)), dtype=np.float64),
                wb=np.zeros(5, dtype=np.float32),
                nsol=np.int32(2),
                ndsol=np.int32(2),
                sol=np.zeros(((3 + 3), 2), dtype=np.float64),
                dsol=np.zeros((5, 2), dtype=np.float64),
                lsol=np.int32(0),
                h=np.zeros((3,2), dtype=np.int32),
                qn=np.zeros(3, dtype=np.float64),
                cutoff=np.float64(0),
                ci=np.zeros((4,3), dtype=np.float64),
                tnmat=np.zeros((4,3), dtype=np.float64),
                big=np.finfo(np.float64).max,
                lci1=np.bool_(False))

        # These tuples contain arrays so we can't assert equality of
        # in a single comparison

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
        X = np.array([[1,2,3],
                      [4,5,6],
                      [10,20,30],
                      [1,10,100],
                      [7,5,3]])
        y = np.array([10,100,1000,5,25])
        tau = None
        actual_params = fit_br.derive_br_params(X, y, tau)
        expected_params = fit_br.BRParams(m=5,
                nn=np.int32(3),
                m5=np.int32(5 + 5),
                n3=np.int32(3 + 3),
                n4=np.int32(3 + 4),
                a=X,
                b=y,
                t=-1,
                toler=np.finfo(np.float64).eps ** (2/3),
                ift=np.int32(1),
                x=np.zeros(3, np.float64),
                e=np.zeros(5, np.float64),
                s=np.zeros(5, dtype=np.int32),
                wa=np.zeros(((5 + 5),(3 + 4)), dtype=np.float64),
                wb=np.zeros(5, dtype=np.float32),
                nsol=np.int32(15),
                ndsol=np.int32(15),
                sol=np.zeros(((3 + 3), 15), dtype=np.float64),
                dsol=np.zeros((5, 15), dtype=np.float64),
                lsol=np.int32(0),
                h=np.zeros((3,15), dtype=np.int32),
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

