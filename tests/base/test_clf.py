import sys
sys.path.append("../../")
from lib.base.testcase import TestPipeline
import torch
import unittest

class case1(TestPipeline):
    def __init__(self) -> None:
        super().__init__(testcase_name = "probing classifier test", test_type = "base")
    def test_case(self):
        from lib.base import clf
        def test_loss():
            self.assertTrue(hasattr(clf.Loss(), "variational"))
            lossnotvar = clf.Loss(variational=False)
            lossvar = clf.Loss(variational=True)

        def test_var_layer():
            varlayer = clf.LinearVariational(in_features = 3, 
                                            out_features = 2, 
                                            parent = clf.KL, 
                                            device = torch.device('cpu'))
            out = varlayer(torch.zeros(1, 3))

        def test_model():
            modelvar = clf.LinearModel(in_size = 3,
                                    hidden_size = 2, 
                                    out_size = 2, 
                                    variational = True, 
                                    device = torch.device('cpu'))
            out = modelvar(torch.zeros(1, 3))
            modelnorvar = clf.LinearModel(in_size = 3,
                                    hidden_size = 2, 
                                    out_size = 2, 
                                    variational = False, 
                                    device = torch.device('cpu'))
            out = modelnorvar(torch.zeros(1, 3))

        def test_loss_backprop():
            modelvar = clf.LinearModel(in_size = 3,
                                    hidden_size = 2, 
                                    out_size = 2, 
                                    variational = True, 
                                    device = torch.device('cpu'))
            lossvar = clf.Loss(variational=True)
            out = modelvar(torch.zeros(1, 3))
            lossvar(torch.zeros((1)).long(), out, model = modelvar).backward()
        #####cases    
        _ = test_loss()
        _ = test_var_layer()
        _ = test_model()
        _ = test_loss_backprop()
        print("tests passed")

    def collect_statistics(self) -> dict:
        def warp_test_suite(testcase_class):
            """Load tests from a specific set of TestCase classes."""
            suite = unittest.TestSuite()
            tests = unittest.defaultTestLoader.loadTestsFromTestCase(testcase_class)
            suite.addTest(tests)
            return suite

        result_value = {"Failures": 0, "Errors": 0, "Skipped": 0, "Test Runs": 0}
        runner = unittest.TextTestRunner()
        TextTestResult = runner.run(warp_test_suite(self))

        # Passes the Result
        result_value["Failures"] += len(TextTestResult.failures)
        result_value["Errors"] += len(TextTestResult.errors)
        result_value["Skipped"] += len(TextTestResult.skipped)
        result_value["Test Runs"] += TextTestResult.testsRun
        return result_value


def main(): _ = case1()

if __name__ == "__main__": main()