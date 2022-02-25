import unittest

class TestPipeline(unittest.TestCase):
    def __init__(self, testcase_name: str, test_type: str) -> None:
        self.type = test_type
        self.name = testcase_name

    def test_case(self):
        raise NotImplementedError("")

    def collect_statistics(self):
        raise NotImplementedError("")
