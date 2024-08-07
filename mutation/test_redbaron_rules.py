# ===========================================================
#
# Mutation labelling program - Test RedBaron
#
# Author: Guillaume Steveny
# Year: 2023 -- 2024
#
# ===========================================================

import unittest
import redbaron_rules as rr
from mutation_rule import NoMatches


class TestRemoveTry(unittest.TestCase):
    """
    Test for the RemoveTry redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RemoveTry("", False, None)

    # A basic tet with a single try-except block
    def test_basic(self):
        code = """
        def fun(n):
            try:
                pass
            except:
                return None"""

        expected = """
        def fun(n):
            pass
            """

        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    # A more complex test where the try contain more than a line
    def test_complex(self):
        code = """
        def fun(n):
            if False:
                return 0
            try:
                try:
                    pass
                except:
                    return 0
            except:
                return 1
            return None"""

        expected = """
        def fun(n):
            if False:
                return 0
            try:
                pass
            except:
                return 0
            return None\n"""

        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    # A more complex test where the try contain more than a line and is inside an indented block
    def test_indented(self):
        code = """
        def fun(n):
            while True:
                try:
                    try:
                        pass
                    except:
                        return 0
                except:
                    return 1"""

        expected = """
        def fun(n):
            while True:
                try:
                    pass
                except:
                    return 0
                """

        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# ------------------------------------------------------------------


class TestOtherVariable(unittest.TestCase):
    """
    Test for the OtherVariable redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.OtherVariable("", False, None)

    # A basic test where the arg is replaced by the other variable
    def test_basic(self):
        code = """
        def fun(n):
            a = n
            return a"""

        expected = """
        def fun(a):
            a = n
            return a\n"""

        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    # A more complex test where the "a" is replaced by the True
    def test_complex(self):
        code = """
        def fun():
            a = 0
            if True:
                if True:
                    if True: 
                        try:
                            pass
                        except Exception as e:
                            pass
            return a"""

        expected = """
        def fun():
            True = 0
            if True:
                if True:
                    if True: 
                        try:
                            pass
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# ------------------------------------------------------------------


class TestValueChange(unittest.TestCase):
    """
    Test of the ValueChange redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.ValueChange("", False, None)
        self.int_10 = rr.ValueChange("", False, None, "value_change i 10")
        self.int_minus_10 = rr.ValueChange("", False, None, "value_change i -10")
        self.float_10 = rr.ValueChange("", False, None, "value_change f 10")
        self.float_minus_10 = rr.ValueChange("", False, None, "value_change f -10")
        self.all_10 = rr.ValueChange("", False, None, "value_change a 10.5")
        self.all_minus_10 = rr.ValueChange("", False, None, "value_change a -10.5")

    # A basic test where the first 0 is replaced by 1
    def test_basic(self):
        code = """
        def fun():
            a = 0
            b = 0.0
            return a"""

        expected = """
        def fun():
            a = 1
            b = 0.0
            return a\n"""

        expected_int_10 = """
        def fun():
            a = 10
            b = 0.0
            return a\n"""

        expected_int_minus_10 = """
        def fun():
            a = -10
            b = 0.0
            return a\n"""

        expected_float_10 = """
        def fun():
            a = 0
            b = 10.0
            return a\n"""

        expected_float_minus_10 = """
        def fun():
            a = 0
            b = -10.0
            return a\n"""

        expected_all_10 = """
        def fun():
            a = 10.5
            b = 0.0
            return a\n"""

        expected_all_minus_10 = """
        def fun():
            a = -10.5
            b = 0.0
            return a\n"""

        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)
        self.assertEqual(self.int_10.apply(code, ("", 0, ""))[2], expected_int_10)
        self.assertEqual(self.int_minus_10.apply(code, ("", 0, ""))[2], expected_int_minus_10)
        self.assertEqual(self.float_10.apply(code, ("", 0, ""))[2], expected_float_10)
        self.assertEqual(self.float_minus_10.apply(code, ("", 0, ""))[2], expected_float_minus_10)
        self.assertEqual(self.all_10.apply(code, ("", 0, ""))[2], expected_all_10)
        self.assertEqual(self.all_minus_10.apply(code, ("", 0, ""))[2], expected_all_minus_10)

    def get_complex(self, rule):
        """
        Method to get the complex example for the following tests

        Args:
            rule: the rule to be applied on the reference code

        Returns:
            the result of the application of the specified rule on the reference code
        """
        code = """
        def fun():
            b = 0.5
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = 1
                        except Exception as e:
                            pass
            return a"""
        return rule.apply(code, ("", 0, ""))[2]

    # Test where only the integer is modified
    def test_complex(self):
        expected = """
        def fun():
            b = 0.5
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = 2
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.rule), expected)

    def test_complex_int_10(self):
        expected = """
        def fun():
            b = 0.5
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = 11
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.int_10), expected)

    def test_complex_int_minus_10(self):
        expected = """
        def fun():
            b = 0.5
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = -9
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.int_minus_10), expected)

    def test_complex_float_10(self):
        expected = """
        def fun():
            b = 10.5
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = 1
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.float_10), expected)

    def test_complex_float_minus_10(self):
        expected = """
        def fun():
            b = -9.5
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = 1
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.float_minus_10), expected)

    def test_complex_all_10(self):
        expected = """
        def fun():
            b = 0.5
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = 11.5
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.all_10), expected)

    def test_complex_all_minus_10(self):
        expected = """
        def fun():
            b = 0.5
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = -9.5
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.all_minus_10), expected)

# -----------------------------------------------


class TestChangeSign(unittest.TestCase):
    """
    Test for the ChangeSign redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.ChangeSign("", False, None)
        self.int_rule = rr.ChangeSign("", False, None, "change_sign i")
        self.float_rule = rr.ChangeSign("", False, None, "change_sign f")
        self.all_rule = rr.ChangeSign("", False, None, "change_sign a")

    def get_basic(self, rule):
        """
        Method to get the basic example for the following tests

        Args:
            rule: the rule to be applied on the reference instance

        Returns:
            the code modified by the application of the rule on the reference code
        """
        code = """
        def fun():
            a = 1
            b = 1.5
            return a"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_basic(self):
        expected = """
        def fun():
            a = -1
            b = 1.5
            return a\n"""

        self.assertEqual(self.get_basic(self.rule), expected)

    def test_basic_int(self):
        expected = """
        def fun():
            a = -1
            b = 1.5
            return a\n"""

        self.assertEqual(self.get_basic(self.int_rule), expected)

    def test_basic_float(self):
        expected = """
        def fun():
            a = 1
            b = -1.5
            return a\n"""

        self.assertEqual(self.get_basic(self.float_rule), expected)

    def test_basic_all(self):
        expected = """
        def fun():
            a = -1
            b = 1.5
            return a\n"""

        self.assertEqual(self.get_basic(self.all_rule), expected)

    def get_complex(self, rule):
        """
        Method to get the complex example for the following tests

        Args:
            rule: the rule to be applied on the reference code

        Returns:
            the modified code after applying the rule on the reference instance
        """
        code = """
        def fun():
            b = (-100.5 + a)
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = (-1000 + 1)
                        except Exception as e:
                            pass
            return a"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_complex(self):
        expected = """
        def fun():
            b = (-100.5 + a)
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = (1000 + 1)
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.rule), expected)

    def test_complex_int(self):
        expected = """
        def fun():
            b = (-100.5 + a)
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = (1000 + 1)
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.int_rule), expected)

    def test_complex_float(self):
        expected = """
        def fun():
            b = (100.5 + a)
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = (-1000 + 1)
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.float_rule), expected)

    def test_complex_all(self):
        expected = """
        def fun():
            b = (-100.5 + a)
            if True:
                if True:
                    if True: 
                        try:
                            a = 0.0
                            b = (1000 + 1)
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.all_rule), expected)

# -----------------------------------------------


class TestRemoveAssignWhile(unittest.TestCase):
    """
    Test for the RemoveAssignWhile redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RemoveAssignWhile("", False, None)
        self.rule_init = rr.RemoveAssignWhile("", False, None, "remove_assign_while i")

    def get_basic(self, rule):
        """
        Method to get the basic example for the following tests

        Args:
            rule: the rule to be applied on this specific instance

        Returns:
            the result of the application of the rule on the reference code
        """
        code = """
        def fun(n):
            i = 0
            while i < n:
                print(i)
                i += 1
            return n"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_basic(self):
        expected = """
        def fun(n):
            i = 0
            while i < n:
                print(i)
            return n\n"""

        self.assertEqual(self.get_basic(self.rule), expected)

    def test_basic_init(self):
        expected = """
        def fun(n):
            while i < n:
                print(i)
                i += 1
            return n\n"""

        self.assertEqual(self.get_basic(self.rule_init), expected)

    def get_complex(self, rule):
        """
        Method to get the complex example for the following tests

        Args:
            rule: the rule to be applied on the reference code

        Returns:
            the result of the modification of the reference code by the specified rule
        """
        code = """
        def fun(n):
            if False:
                return None
            b = 0.5
            a = 0
            m = 0
            while a < n*b:
                while a < m:
                    if True: 
                        try:
                            if False:
                                return None
                            a = 0.0
                            b = 0
                            m += 1
                            pass
                        except Exception as e:
                            pass
            return a"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_complex(self):
        expected = """
        def fun(n):
            if False:
                return None
            b = 0.5
            a = 0
            m = 0
            while a < n*b:
                while a < m:
                    if True: 
                        try:
                            if False:
                                return None
                            m += 1
                            pass
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.rule), expected)

    def test_complex_init(self):
        expected = """
        def fun(n):
            if False:
                return None
            a = 0
            m = 0
            while a < n*b:
                while a < m:
                    if True: 
                        try:
                            if False:
                                return None
                            a = 0.0
                            b = 0
                            m += 1
                            pass
                        except Exception as e:
                            pass
            return a\n"""

        self.assertEqual(self.get_complex(self.rule_init), expected)

# -----------------------------------------------


class TestWithToOpen(unittest.TestCase):
    """
    Test for the WithToOpen redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.WithToOpen("", False, None)
        self.rule_close = rr.WithToOpen("", False, None, "with_to_open c")

    def get_basic(self, rule):
        """
        Method to get the basic example for the following tests

        Args:
            rule: the rule to be applied on the instance generated for these tests

        Returns:
            the modified code with respect to the specified rule and the reference code
        """
        code = """
        def fun(n):
            with open("test", "r") as file:
                text = file.read()"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_basic(self):
        expected = """
        def fun(n):
            file = open("test", "r")
            text = file.read()"""

        self.assertEqual(self.get_basic(self.rule), expected)

    def test_basic_close(self):
        expected = """
        def fun(n):
            file = open("test", "r")
            text = file.read()
            file.close()
            """

        self.assertEqual(self.get_basic(self.rule_close), expected)

    def get_complex(self, rule):
        """
        Method to get the complex example for the following tests

        Args:
            rule: the rule to be applied on the reference code

        Returns:
            the modified code after application of the rule
        """
        code = """
        def func():
            with open("other", "w") as other:
                other.write(text)
                if test == "help":
                    return None
            with open("test", "r") as file:
                text = file.read()
            return text"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_complex(self):
        expected = """
        def func():
            other = open("other", "w")
            other.write(text)
            if test == "help":
                return None
            with open("test", "r") as file:
                text = file.read()
            return text\n"""

        self.assertEqual(self.get_complex(self.rule), expected)

    def test_complex_close(self):
        expected = """
        def func():
            other = open("other", "w")
            other.write(text)
            if test == "help":
                return None
        
            other.close()
            with open("test", "r") as file:
                text = file.read()
            return text\n"""

        self.assertEqual(self.get_complex(self.rule_close), expected)

    def get_multiple_contexts(self, rule):
        """
        Method to get an example where the with has multiple contexts to be transformed

        Arg:
            rule: the rule to be applied on the instance represented here

        Returns:
            the modified code after applying the rule on the reference code
        """
        code = """
        def func():
            with open("other", "w") as other, open("file", "r") as file:
                text = file.read()
                other.write(text)
                if test == "help":
                    return None
            return text"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_multiple(self):
        expected = """
        def func():
            file = open("file", "r")
            other = open("other", "w")
            text = file.read()
            other.write(text)
            if test == "help":
                return None
            return text\n"""
        self.assertEqual(self.get_multiple_contexts(self.rule), expected)

    def test_multiple_close(self):
        expected = """
        def func():
            file = open("file", "r")
            other = open("other", "w")
            text = file.read()
            other.write(text)
            if test == "help":
                return None
        
            other.close()
            file.close()
            return text\n"""
        self.assertEqual(self.get_multiple_contexts(self.rule_close), expected)

# -----------------------------------------------


class TestRemoveReturn(unittest.TestCase):
    """
    Test of the RemoveReturn redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RemoveReturn("", False, None)

    def test_basic(self):
        code = """
        def fun(n):
            a = n * 2
            return a
            pass"""

        expected = """
        def fun(n):
            a = n * 2
                        
            pass\n"""

        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def func():
            with open("other", "w") as other:
                other.write(text)
                if test == "help":
                    if False:
                        pass
                    return None
            with open("test", "r") as file:
                text = file.read()
            return text"""

        expected = """
        def func():
            with open("other", "w") as other:
                other.write(text)
                if test == "help":
                    if False:
                        pass
                                        
            with open("test", "r") as file:
                text = file.read()
            return text\n"""

        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# -----------------------------------------------


class TestForRange(unittest.TestCase):
    """
    Test of the ForRange redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.ForRange("", False, None)

    def test_basic(self):
        code = """
        def func():
            for i in s:
                print(i)"""

        expected = """
        def func():
            for i in range(s):
                print(i)\n"""

        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def func():
            for i in range(10):
                print(i)
            for i in [1,2,3]:
                print(i)"""

        expected = """
        def func():
            for i in range(10):
                print(i)
            for i in range([1,2,3]):
                print(i)\n"""

        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# -----------------------------------------------


class TestRemoveNestedLoop(unittest.TestCase):
    """
    Test for the RemoveNestedLoop redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RemoveNestedLoop("", False, None)
        self.rule_delete = rr.RemoveNestedLoop("", False, None, "remove_nested_loop d")

    def get_basic(self, rule):
        """
        Method to get the basic example for the following tests

        Args:
            rule: the rule to be applied on this reference instance code

        Returns:
            the modified code after applying the specified rule on the reference code
        """
        code = """
        def fun():
            for i in range(10):
                print(i)
            return True"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_basic(self):
        expected = """
        def fun():
            print(i)
            return True\n"""

        self.assertEqual(self.get_basic(self.rule), expected)

    def test_basic_delete(self):
        expected = """
        def fun():
            
            return True\n"""

        self.assertEqual(self.get_basic(self.rule_delete), expected)

    def get_complex(self, rule):
        """
        Method to get the complex example for the following tests

        Args:
            rule: the rule to be applied on this reference code

        Returns:
            the modified code after applying the rule on the reference code
        """
        code = """
        def fun():
            if False:
                return False
            for i in range(10):
                for j in range(100):
                    print(j)
                    if True:
                        print(i)
                print(i)
                pass
            return True"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_complex(self):
        expected = """
        def fun():
            if False:
                return False
            for j in range(100):
                print(j)
                if True:
                    print(i)
            print(i)
            pass
            return True\n"""

        self.assertEqual(self.get_complex(self.rule), expected)

    def test_complex_delete(self):
        expected = """
        def fun():
            if False:
                return False
            
            return True\n"""

        self.assertEqual(self.get_complex(self.rule_delete), expected)

    def get_in_if(self, rule):
        """
        Method to get an example where the modification is made inside an if-block

        Args:
            rule: the rule to be applied on the reference code in this particular example

        Returns:
            the modified code after applying the rule on the reference code
        """
        code = """
        def fun():
            if False:
                return False
            if i < 10:
                for j in range(100):
                    print(j)
                    if True:
                        print(i)
                print(i)
                pass
            return True"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_in_if(self):
        expected = """
        def fun():
            if False:
                return False
            if i < 10:
                print(j)
                if True:
                    print(i)
                
                print(i)
                pass
            return True\n"""

        self.assertEqual(self.get_in_if(self.rule), expected)

    def test_in_if_delete(self):
        expected = """
        def fun():
            if False:
                return False
            if i < 10:
                
                print(i)
                pass
            return True\n"""

        self.assertEqual(self.get_in_if(self.rule_delete), expected)

# -----------------------------------------------


class TestRemoveIfElseBlock(unittest.TestCase):
    """
    Test for the RemoveIfElseBlock redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RemoveIfElseBlock("", False, None)
        self.rule_else = rr.RemoveIfElseBlock("", False, None, "remove_if_else e")

    def get_basic(self, rule):
        """
        Method to get the basic example for the following tests

        Args:
            rule: the rule to be applied on the reference code

        Returns:
            the result of the application of the rule on the reference code
        """
        code = """
        def fun():
            if True:
                print(0)
            else:
                print(1)"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_basic(self):
        expected = """
        def fun():
            print(0)\n"""
        self.assertEqual(self.get_basic(self.rule), expected)

    def test_basic_else(self):
        expected = """
        def fun():
            print(1)\n"""
        self.assertEqual(self.get_basic(self.rule_else), expected)

    def get_complex(self, rule):
        """
        Method to get the complex example for the following tests

        Args:
            rule: the rule to be applied on the reference code

        Returns:
            the result of the application of the rule on the reference code
        """
        code = """
        def fun():
            if True:
                out = 0
                for i in range(10):
                    out += i
                    if False:
                        return 0
                print(0)
            elif False:
                pass
            else:
                out = 0
                while out < 10:
                    out += 1
                    if True:
                        return 1
                print(1)"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_complex(self):
        expected = """
        def fun():
            out = 0
            for i in range(10):
                out += i
                if False:
                    return 0
            print(0)\n"""
        self.assertEqual(self.get_complex(self.rule), expected)

    def test_complex_else(self):
        expected = """
        def fun():
            out = 0
            while out < 10:
                out += 1
                if True:
                    return 1
            print(1)\n"""
        self.assertEqual(self.get_complex(self.rule_else), expected)

    def get_in_try(self, rule):
        """
        Method to get an example where the match is in a try

        Args:
            rule: the to be applied on the reference code

        Returns:
            the result of the application of the rule on the reference code
        """
        code = """
        def fun():
            try:
                if True:
                    out = 0
                    for i in range(10):
                        out += i
                        if False:
                            return 0
                    print(0)
                elif False:
                    pass
                else:
                    out = 0
                    while out < 10:
                        out += 1
                        if True:
                            return 1
                    print(1)
            except:
                return None"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_in_try(self):
        expected = """
        def fun():
            try:
                out = 0
                for i in range(10):
                    out += i
                    if False:
                        return 0
                print(0)
            except:
                return None\n"""
        self.assertEqual(self.get_in_try(self.rule), expected)

    def test_in_try_else(self):
        expected = """
        def fun():
            try:
                out = 0
                while out < 10:
                    out += 1
                    if True:
                        return 1
                print(1)
            except:
                return None\n"""
        self.assertEqual(self.get_in_try(self.rule_else), expected)

# -----------------------------------------------


class TestCompletelyRemoveCdtBlock(unittest.TestCase):
    """
    Test for the CompletelyRemoveCdtBlock redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.CompletelyRemoveCdtBlock("", False, None)
        self.if_rule = rr.CompletelyRemoveCdtBlock("", False, None, "suppress_if if")
        self.elif_rule = rr.CompletelyRemoveCdtBlock("", False, None, "suppress_if elif")
        self.else_rule = rr.CompletelyRemoveCdtBlock("", False, None, "suppress_if else")

    def get_basic(self, rule):
        """
        Method to get the basic example for the following tests

        Args:
            rule: the rule to be applied on the reference code

        Returns:
            the result of the application of the rule on the reference code
        """
        code = """
        def fun(a, b):
            print(a,b)
            if a > b:
                return a
            elif b < a:
                return b
            else:
                return 0"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_basic(self):
        expected = """
        def fun(a, b):
            print(a,b)\n\n"""
        self.assertEqual(self.get_basic(self.rule), expected)

    def test_basic_if(self):
        expected = """
        def fun(a, b):
            print(a,b)
            
            elif b < a:
                return b
            else:
                return 0\n"""
        self.assertEqual(self.get_basic(self.if_rule), expected)

    def test_basic_elif(self):
        expected = """
        def fun(a, b):
            print(a,b)
            if a > b:
                return a
            
            else:
                return 0\n"""
        self.assertEqual(self.get_basic(self.elif_rule), expected)

    def test_basic_else(self):
        expected = """
        def fun(a, b):
            print(a,b)
            if a > b:
                return a
            elif b < a:
                return b
            \n"""
        self.assertEqual(self.get_basic(self.else_rule), expected)

    def get_complex(self, rule):
        """
        Method to get the complex example for the following tests

        Args:
            rule: the rule to be applied on the reference code

        Returns:
            the result of the application of the rule
        """
        code = """
        def fun(a):
            if a > 0:
                while True:
                    if a < 0:
                        a -= a
                        if True:
                            print(a)
                    elif a > 0:
                        a += a
                        if False:
                            return a
                            if False:
                                print(a)
                        else:
                            if a > 0:
                                print(a)
                        a *= 2
                        pass
                    elif False:
                        return False
                    print(2*a)
                    pass
                print(a)
            else:
                print(False)
            while False:
                print(a)
            return None"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_complex(self):
        expected = """
        def fun(a):\n
            while False:
                print(a)
            return None\n"""
        self.assertEqual(self.get_complex(self.rule), expected)

    def test_complex_if(self):
        expected = """
        def fun(a):
            
            else:
                print(False)
            while False:
                print(a)
            return None\n"""
        self.assertEqual(self.get_complex(self.if_rule), expected)

    def test_complex_elif(self):
        expected = """
        def fun(a):
            if a > 0:
                while True:
                    if a < 0:
                        a -= a
                        if True:
                            print(a)
                    
                    elif False:
                        return False
                    print(2*a)
                    pass
                print(a)
            else:
                print(False)
            while False:
                print(a)
            return None\n"""
        self.assertEqual(self.get_complex(self.elif_rule), expected)

    def test_complex_else(self):
        expected = """
        def fun(a):
            if a > 0:
                while True:
                    if a < 0:
                        a -= a
                        if True:
                            print(a)
                    elif a > 0:
                        a += a
                        if False:
                            return a
                            if False:
                                print(a)
                        
                        a *= 2
                        pass
                    elif False:
                        return False
                    print(2*a)
                    pass
                print(a)
            else:
                print(False)
            while False:
                print(a)
            return None\n"""
        self.assertEqual(self.get_complex(self.else_rule), expected)

    def get_missing(self, rule):
        """
        Method to get the example with missing match for the following tests

        Args:
            rule: the rule to be applied on the reference code

        Returns:
            the result of the rule application on the code without match
        """
        code = """
        def fun():
            while False:
                print(True)
            return False"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_missing_all(self):
        rules = [self.rule, self.if_rule, self.elif_rule, self.else_rule]
        for rule in rules:
            with self.assertRaises(NoMatches):
                self.get_missing(rule)

# ----------------------------------


class TestRevertSlice(unittest.TestCase):
    """
    Test for the RevertSlice redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rules = [rr.RevertSlice("", False, None),
                      rr.RevertSlice("", False, None, "revert_slice l"),
                      rr.RevertSlice("", False, None, "revert_slice u"),
                      rr.RevertSlice("", False, None, "revert_slice a")]
        self.basic_configs = ["[:3]", "[1:]", "[1:3]",
                              "[:-3]", "[-1:]", "[-1:-3]",
                              "[:3:2]", "[1::2]", "[1:3:2]"]
        self.complex_configs = ["[:b+c]", "[a+c:]", "[a+c:b+c]",
                                "[:-b+c]", "[-a+c:]", "[-a+c:-b+c]",
                                "[0:int('1')]",
                                "[:b+c:c]", "[a+c::c]", "[a+c:b+c:c]"]

    def get_basic(self, config, rule):
        """
        Method to get the basic example for the test of this TestCase with a specific configuration

        Args:
            config: the patch to be inserted inside the basic code
            rule: the rule to be applied on the reference code structure

        Returns:
            the modified code with respect to the specific rule and containing the config snippet
        """
        code = self.get_basic_config(config)
        return rule.apply(code, ("", 0, ""))[2]

    def get_basic_config(self, config):
        """
        Create a code snippet with a particular config

        Args:
            config: a string to be contained in the generated instance

        Returns:
            the newly constructed code example
        """
        code = f"""
        def fun(l):
            return l{config}"""
        return code

    def test_basic(self):
        rule = self.rules[0]
        expecteds = ["[3:]", "[:1]", "[3:1]",
                     "[-3:]", "[:-1]", "[-3:-1]",
                     "[3::2]", "[:1:2]", "[3:1:2]"]
        for c, e in zip(self.basic_configs, expecteds):
            self.assertEqual(self.get_basic(c, rule), self.get_basic_config(e)+"\n")

    def test_basic_lower(self):
        rule = self.rules[1]
        expecteds = ["[3:]", "[:-1]", "[3:-1]",
                     "[-3:]", "[:1]", "[-3:1]",
                     "[3::2]", "[:-1:2]", "[3:-1:2]"]
        for c, e in zip(self.basic_configs, expecteds):
            self.assertEqual(self.get_basic(c, rule), self.get_basic_config(e)+"\n")

    def test_basic_upper(self):
        rule = self.rules[2]
        expecteds = ["[-3:]", "[:1]", "[-3:1]",
                     "[3:]", "[:-1]", "[3:-1]",
                     "[-3::2]", "[:1:2]", "[-3:1:2]"]
        for c, e in zip(self.basic_configs, expecteds):
            self.assertEqual(self.get_basic(c, rule), self.get_basic_config(e)+"\n")

    def test_basic_all(self):
        rule = self.rules[3]
        expecteds = ["[-3:]", "[:-1]", "[-3:-1]",
                     "[3:]", "[:1]", "[3:1]",
                     "[-3::2]", "[:-1:2]", "[-3:-1:2]"]
        for c, e in zip(self.basic_configs, expecteds):
            self.assertEqual(self.get_basic(c, rule), self.get_basic_config(e)+"\n")

    def get_complex(self, config, rule):
        """
        Method to get the complex example for the test of this TestCase with a specific configuration

        Args:
            config: the patch to be inserted inside the basic code
            rule: the rule to be applied on the reference code structure

        Returns:
            the modified code with respect to the specific rule and containing the config snippet
        """
        code = self.get_complex_config(config)
        return rule.apply(code, ("", 0, ""))[2]

    def get_complex_config(self, config):
        """
        Create a code snippet with a particular config

        Args:
            config: a string to be contained in the generated instance

        Returns:
            the newly constructed code example
        """
        code = f"""
        def fun(l):
            if True:
                while i < 10:
                    print(l{config})
            return 0"""
        return code

    def test_complex(self):
        rule = self.rules[0]
        expecteds = ["[b+c:]", "[:a+c]", "[b+c:a+c]",
                     "[-b+c:]", "[:-a+c]", "[-b+c:-a+c]",
                     "[int('1'):0]",
                     "[b+c::c]", "[:a+c:c]", "[b+c:a+c:c]"]
        for c, e in zip(self.complex_configs, expecteds):
            self.assertEqual(self.get_complex(c, rule), self.get_complex_config(e)+"\n")

    def test_complex_lower(self):
        rule = self.rules[1]
        expecteds = ["[b+c:]", "[:-a+c]", "[b+c:-a+c]",
                     "[-b+c:]", "[:a+c]", "[-b+c:a+c]",
                     "[int('1'):-0]",
                     "[b+c::c]", "[:-a+c:c]", "[b+c:-a+c:c]"]
        for c, e in zip(self.complex_configs, expecteds):
            self.assertEqual(self.get_complex(c, rule), self.get_complex_config(e) + "\n")

    def test_complex_upper(self):
        rule = self.rules[2]
        expecteds = ["[-b+c:]", "[:a+c]", "[-b+c:a+c]",
                     "[b+c:]", "[:-a+c]", "[b+c:-a+c]",
                     "[-int('1'):0]",
                     "[-b+c::c]", "[:a+c:c]", "[-b+c:a+c:c]"]
        for c, e in zip(self.complex_configs, expecteds):
            self.assertEqual(self.get_complex(c, rule), self.get_complex_config(e) + "\n")

    def test_complex_all(self):
        rule = self.rules[3]
        expecteds = ["[-b+c:]", "[:-a+c]", "[-b+c:-a+c]",
                     "[b+c:]", "[:a+c]", "[b+c:a+c]",
                     "[-int('1'):-0]",
                     "[-b+c::c]", "[:-a+c:c]", "[-b+c:-a+c:c]"]
        for c, e in zip(self.complex_configs, expecteds):
            self.assertEqual(self.get_complex(c, rule), self.get_complex_config(e) + "\n")

# ----------------------------------


class TestSuppressPartSlice(unittest.TestCase):
    """
    Test for the SuppressPartSlice redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rules = [rr.SuppressPartSlice("", False, None),
                      rr.SuppressPartSlice("", False, None, "suppress_part_slice l"),
                      rr.SuppressPartSlice("", False, None, "suppress_part_slice u"),
                      rr.SuppressPartSlice("", False, None, "suppress_part_slice s"),
                      rr.SuppressPartSlice("", False, None, "suppress_part_slice lu"),
                      rr.SuppressPartSlice("", False, None, "suppress_part_slice us"),
                      rr.SuppressPartSlice("", False, None, "suppress_part_slice ls"),
                      rr.SuppressPartSlice("", False, None, "suppress_part_slice lus")]

    def get_basic(self, config, rule):
        """
        Method to get the basic example for the test of this TestCase with a specific configuration

        Args:
            config: the patch to be inserted inside the basic code
            rule: the rule to be applied on the reference code structure

        Returns:
            the modified code with respect to the specific rule and containing the config snippet
        """
        code = self.get_basic_config(config)
        return rule.apply(code, ("", 0, ""))[2]

    def get_basic_config(self, config):
        """
        Create a code snippet with a particular config

        Args:
            config: a string to be contained in the generated instance

        Returns:
            the newly constructed code example
        """
        code = f"""
        def fun(l):
            return l{config}"""
        return code

    def test_basic(self):
        rule = self.rules[0]
        configs = ["[1:3:2]", "[a+1:b+3:c+2]"]
        expecteds = ["[:]", "[:]"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.get_basic(c, rule), self.get_basic_config(e)+"\n")

    def test_basic_lower(self):
        rule = self.rules[1]
        configs = ["[1:]", "[1:3]",
                   "[-1:]", "[-1:-3]",
                   "[1::2]", "[1:3:2]",
                   "[a+1:]", "[a+1:b+3]",
                   "[a-1:]", "[a-1:b-3]",
                   "[a+1::c+2]", "[a+1:b+3:c+2]"]
        expecteds = ["[:]", "[:3]",
                     "[:]", "[:-3]",
                     "[::2]", "[:3:2]",
                     "[:]", "[:b+3]",
                     "[:]", "[:b-3]",
                     "[::c+2]", "[:b+3:c+2]"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.get_basic(c, rule), self.get_basic_config(e)+"\n")

    def test_basic_upper(self):
        rule = self.rules[2]
        configs = ["[:3]", "[1:3]",
                   "[:-3]", "[-1:-3]",
                   "[:3:2]", "[1:3:2]",
                   "[:b+3]", "[a+1:b+3]",
                   "[:b-3]", "[a-1:b-3]",
                   "[:b+3:c+2]", "[a+1:b+3:c+2]"]
        expecteds = ["[:]", "[1:]",
                     "[:]", "[-1:]",
                     "[::2]", "[1::2]",
                     "[:]", "[a+1:]",
                     "[:]", "[a-1:]",
                     "[::c+2]", "[a+1::c+2]"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.get_basic(c, rule), self.get_basic_config(e)+"\n")

    def test_basic_step(self):
        rule = self.rules[3]
        configs = ["[:3:2]", "[1::2]", "[1:3:2]",
                   "[:b+3:c+2]", "[a+1::c+2]", "[a+1:b+3:c+2]"]
        expecteds = ["[:3]", "[1:]", "[1:3]",
                     "[:b+3]", "[a+1:]", "[a+1:b+3]"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.get_basic(c, rule), self.get_basic_config(e)+"\n")

    def test_basic_lu(self):
        rule = self.rules[4]
        configs = ["[1:3]", "[-1:-3]", "[1:3:2]", "[a+1:b+3]", "[a-1:b-3]", "[a+1:b+3:c+2]"]
        expecteds = ["[:]", "[:]", "[::2]", "[:]", "[:]", "[::c+2]"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.get_basic(c, rule), self.get_basic_config(e)+"\n")

    def test_basic_us(self):
        rule = self.rules[5]
        configs = ["[:3:2]", "[1:3:2]",
                   "[:b+3:c+2]", "[a+1:b+3:c+2]"]
        expecteds = ["[:]", "[1:]",
                     "[:]", "[a+1:]"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.get_basic(c, rule), self.get_basic_config(e)+"\n")

    def test_basic_ls(self):
        rule = self.rules[6]
        configs = ["[1::2]", "[1:3:2]",
                   "[a+1::c+2]", "[a+1:b+3:c+2]"]
        expecteds = ["[:]", "[:3]",
                     "[:]", "[:b+3]"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.get_basic(c, rule), self.get_basic_config(e)+"\n")

    def test_basic_lus(self):
        rule = self.rules[7]
        configs = ["[1:3:2]", "[a+1:b+3:c+2]"]
        expecteds = ["[:]", "[:]"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.get_basic(c, rule), self.get_basic_config(e)+"\n")

    def get_missing(self, config, rule):
        """
        Method to get an example with incorrect match for the test of this TestCase with a specific configuration

        Args:
            config: the patch to be inserted inside the basic code
            rule: the rule to be applied on the reference code structure

        Returns:
            the modified code with respect to the specific rule and containing the config snippet
        """
        code = self.get_basic_config(config)
        return rule.apply(code, ("", 0, ""))[2]

    def test_missing(self):
        rule = self.rules[0]
        configs = ["[1:]", "[:3]", "[1:3]",
                   "[-1:]", "[:-3]", "[-1:-3]",
                   "[1::2]", "[:3:2]",
                   "[a+1:]", "[:b+3]", "[a+1:b+3]",
                   "[a-1:]", "[:b-3]", "[a-1:b-3]",
                   "[a+1::c+2]", "[:b+3:c+2]"]
        for c in configs:
            with self.assertRaises(NoMatches):
                self.get_missing(c, rule)

    def test_missing_lower(self):
        rule = self.rules[1]
        configs = ["[:3]", "[:-3]", "[:3:2]", "[:b+3]", "[:b-3]", "[:b+3:c+2]"]
        for c in configs:
            with self.assertRaises(NoMatches):
                self.get_missing(c, rule)

    def test_missing_upper(self):
        rule = self.rules[2]
        configs = ["[1:]", "[-1:]", "[1::2]", "[a+1:]", "[a-1:]", "[a+1::c+2]"]
        for c in configs:
            with self.assertRaises(NoMatches):
                self.get_missing(c, rule)

    def test_missing_step(self):
        rule = self.rules[3]
        configs = ["[1:]", "[:3]", "[1:3]",
                   "[-1:]", "[:-3]", "[-1:-3]",
                   "[a+1:]", "[:b+3]", "[a+1:b+3]",
                   "[a-1:]", "[:b-3]", "[a-1:b-3]"]
        for c in configs:
            with self.assertRaises(NoMatches):
                self.get_missing(c, rule)

    def test_missing_us(self):
        rule = self.rules[5]
        configs = ["[1:]", "[:3]", "[1:3]",
                   "[-1:]", "[:-3]", "[-1:-3]",
                   "[1::2]",
                   "[a+1:]", "[:b+3]", "[a+1:b+3]",
                   "[a-1:]", "[:b-3]", "[a-1:b-3]",
                   "[a+1::c+2]"]
        for c in configs:
            with self.assertRaises(NoMatches):
                self.get_missing(c, rule)

    def test_missing_ls(self):
        rule = self.rules[6]
        configs = ["[1:]", "[:3]", "[1:3]",
                   "[-1:]", "[:-3]", "[-1:-3]",
                   "[:3:2]",
                   "[a+1:]", "[:b+3]", "[a+1:b+3]",
                   "[a-1:]", "[:b-3]", "[a-1:b-3]",
                   "[:b+3:c+2]"]
        for c in configs:
            with self.assertRaises(NoMatches):
                self.get_missing(c, rule)

    def test_missing_lus(self):
        rule = self.rules[7]
        configs = ["[1:]", "[:3]", "[1:3]",
                   "[-1:]", "[:-3]", "[-1:-3]",
                   "[1::2]", "[:3:2]",
                   "[a+1:]", "[:b+3]", "[a+1:b+3]",
                   "[a-1:]", "[:b-3]", "[a-1:b-3]",
                   "[a+1::c+2]", "[:b+3:c+2]"]
        for c in configs:
            with self.assertRaises(NoMatches):
                self.get_missing(c, rule)

# ----------------------------------


class TestRemoveArgFromCall(unittest.TestCase):
    """
    Test for the RemoveArgFromCall redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RemoveArgFromCall("", False, None)

    def test_basic(self):
        code = """
        def fun(n):
            out = 0
            for i in range(0, n):
                out += i
            return out"""
        expected = """
        def fun(n):
            out = 0
            for i in range(n):
                out += i
            return out\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun(a, b, x):
            value = (lambda x: a*x + b - math.ceil(math.floor(3.1)))(x)
            for i in range((lambda x: a*x + b - math.ceil(math.floor(3.1)))(x)):
                value -= i
            return value"""
        expected = """
        def fun(a, b, x):
            value = (lambda x: a*x + b - math.ceil())(x)
            for i in range((lambda x: a*x + b - math.ceil(math.floor(3.1)))(x)):
                value -= i
            return value\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_missing(self):
        code = """
        def fun(n):
            print(n)
            return n"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

# ----------------------------------


class TestRemoveCall(unittest.TestCase):
    """
    Test for theRemoveCall redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RemoveCall("", False, None)

    def test_basic(self):
        code = """
        def fun(n):
            out = 0
            for i in range(0, n):
                out += i
            return out"""
        expected = """
        def fun(n):
            out = 0
            for i in range:
                out += i
            return out\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun(a, b, x):
            value = (lambda x: a*x + b - math.ceil(math.floor(3.1)))(x)
            for i in range((lambda x: a*x + b - math.ceil(math.floor(3.1)))(x)):
                value -= i
            return value"""
        expected = """
        def fun(a, b, x):
            value = (lambda x: a*x + b - math.ceil)(x)
            for i in range((lambda x: a*x + b - math.ceil(math.floor(3.1)))(x)):
                value -= i
            return value\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_missing(self):
        code = """
        def fun(n):
            print(n)
            return n"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

# ----------------------------------


class TestRemoveSelf(unittest.TestCase):
    """
    Test for the RemoveSelf redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RemoveSelf("", False, None)

    def test_basic(self):
        code = """
        class Test:
            def __init__(self, val):
                self.value = val"""
        expected = """
        class Test:
            def __init__(val):
                self.value = val\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_single_arg(self):
        code = """
        class Test:
            def __init__(self):
                pass"""
        expected = """
        class Test:
            def __init__():
                pass\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        class Test:
            def fun(a, b, x):
                value = (lambda x: a*x + b - math.ceil(math.floor(self.value)))(x)
                for i in range((lambda x: a*x + b - math.ceil(math.floor(self.value)))(x)):
                    value -= i
                return value"""
        expected = """
        class Test:
            def fun(a, b, x):
                value = (lambda x: a*x + b - math.ceil(math.floor(value)))(x)
                for i in range((lambda x: a*x + b - math.ceil(math.floor(self.value)))(x)):
                    value -= i
                return value\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_no_class(self):
        code = """
        def fun(n):
            print(self, n)
            return n"""
        expected = """
        def fun(n):
            print(n)
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_missing(self):
        code = """
        def fun(n):
            print(n)
            return n"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

# ----------------------------------


class TestListToTuple(unittest.TestCase):
    """
    Test for the ListToTuple redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.ListToTuple("", False, None)

    def test_basic(self):
        code = """
        def fun():
            l = {v}
            return l"""
        expected = """
        def fun():
            l = {v}
            return l\n"""
        configs = [("[1,2,3]", "(1,2,3)"), ("[1, 2, 3]", "(1, 2, 3)"), ("[]", "()")]
        for c, e in configs:
            self.assertEqual(self.rule.apply(code.format(v=c), ("", 0, ""))[2], expected.format(v=e))

    def test_complex(self):
        code = """
        def fun():
            if True:
                l = (((k[0] for k in enumerate(range(10)) if k > 0), ([j,i] for j, i in zip(range(10), range(10)) if i % 2 == 0)))
            else:
                l = None
            return l"""
        expected = """
        def fun():
            if True:
                l = (((k[0] for k in enumerate(range(10)) if k > 0), ((j,i) for j, i in zip(range(10), range(10)) if i % 2 == 0)))
            else:
                l = None
            return l\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# ----------------------------------


class TestChangeComparison(unittest.TestCase):
    """
    Test for the ChangeComparison redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.ChangeComparison("", False, None)

    def get_basic_config(self, config, rule):
        """
        Method to get the modification of a reference with a specific configuration

        Args:
            config: the configuration to be used for the reference code
            rule: the rule to be applied on the reference code

        Returns:
            the modified reference code with respect to the specified rule
        """
        code = f"""
        def fun(a, b):
            return a {config} b"""
        return rule.apply(code, ("", 0, ""))[2]

    def test_basic(self):
        configs = ["==", "<=", ">=", "!="]
        for c in configs:
            self.assertTrue(c not in self.get_basic_config(c, self.rule))

# ----------------------------------


class TestIdentity(unittest.TestCase):
    """
    Test for the Identity redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.Identity("", False, None)

    def test_basic(self):
        code = """
        def fun(a, b):
            if a < b:
                while a < b:
                    print(a)
                    a += 1
                return b
            else:
                return b"""
        expected = """
        def fun(a, b):
            if a < b:
                while a < b:
                    print(a)
                    a += 1
                return b
            else:
                return b"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        class Test:
            def fun(a, b, x):
                value = (lambda x: a*x + b - math.ceil(math.floor(self.value)))(x)
                for i in range((lambda x: a*x + b - math.ceil(math.floor(self.value)))(x)):
                    value -= i
                return value"""
        expected = """
        class Test:
            def fun(a, b, x):
                value = (lambda x: a*x + b - math.ceil(math.floor(self.value)))(x)
                for i in range((lambda x: a*x + b - math.ceil(math.floor(self.value)))(x)):
                    value -= i
                return value"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# ----------------------------------


class TestMisplacedReturn(unittest.TestCase):
    """
    Test for the MisplacedReturn redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.MisplacedReturn("", False, None)

    def test_basic(self):
        code = """
        def fun(n):
            print(n)
            return n"""
        expected = """
        def fun(n):
            return n
            print(n)\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun(n):
            try:
                def inner(n):
                    if True:
                        def in_inner(n):
                            class Inner:
                                def fct(n):
                                    print(a)
                                    print(b)
                                    if True:
                                        print(True)
                                        while True:
                                            print(a)
                                            print(b)
                                    return n
                            return Inner
                        return n
                    return n
                return n
            except:
                pass"""
        expected = """
        def fun(n):
            try:
                def inner(n):
                    if True:
                        def in_inner(n):
                            class Inner:
                                def fct(n):
                                    return n
                                    print(a)
                                    print(b)
                                    if True:
                                        print(True)
                                        while True:
                                            print(a)
                                            print(b)
                                    
                            return Inner
                        return n
                    return n
                return n
            except:
                pass\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_only_return(self):
        code = """
        def fun(n):
            return n"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

# ----------------------------------


class TestAssignToComparison(unittest.TestCase):
    """
    Test for the AssignToComparison redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.AssignToComparison("", False, None)

    def test_basic(self):
        code = """
        def fun(n):
            m = n"""
        expected = """
        def fun(n):
            m == n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun(a, b):
            if a == b or b <= a or a >= b:
                a += b
                a /= b
                def f(x, y, z, w, s = 0):
                    val = x * y * z + s
                    return val
            return a"""
        expected = """
        def fun(a, b):
            if a == b or b <= a or a >= b:
                a += b
                a /= b
                def f(x, y, z, w, s = 0):
                    val == x * y * z + s
                    return val
            return a\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# ----------------------------------


class TestComparisonToAssign(unittest.TestCase):
    """
    Test for the ComparisonToAssign redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.ComparisonToAssign("", False, None)

    def test_basic(self):
        code = """
        def fun(m, n):
            return m == n"""
        expected = """
        def fun(m, n):
            return m = n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun(a, b):
            if a >= b or b <= a:
                a += b
                a /= b
                def f(x, y, z, w, s = 0):
                    val = x * y * z + s
                    return val == s
            return a"""
        expected = """
        def fun(a, b):
            if a >= b or b <= a:
                a += b
                a /= b
                def f(x, y, z, w, s = 0):
                    val = x * y * z + s
                    return val = s
            return a\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# ----------------------------------


class TestChangeOperandOrder(unittest.TestCase):
    """
    Test for the ChangeOperandOrder redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.ChangeOperandOrder("", False, None)

    def get_basic_config(self, config):
        """
        Method to get a configured reference code
        Args:
            config: the string patch to configure the reference code

        Returns:
            the configured reference code
        """
        code = f"""
        def fun(a1, b1):
            return {config}"""
        return code

    def test_basic(self):
        configs = ["a1 == b1", "a1 != b1", "a1 and b1", "a1 or b1", "a1 ^ b1", "a1 | b1",
                   "a1 & b1", "a1 + b1", "a1 * b1", "a1 @ b1"]
        expecteds = ["b1 == a1", "b1 != a1", "b1 and a1", "b1 or a1", "b1 ^ a1", "b1 | a1",
                     "b1 & a1", "b1 + a1", "b1 * a1", "b1 @ a1"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_basic_config(c), ("", 0, ""))[2], self.get_basic_config(e)+"\n")

    def test_complex_combination(self):
        configs = [("==", "=="), ("!=", "=="), ("==", "!="), ("!=", "!=")]
        for c1, c2 in configs:
            for sa, sb in [("", ""), ("-", "-"), ("-", ""), ("", "-")]:
                config = f"{sa}a1 {c1} {sb}b2 {c2} c3"
                expected = f"{sb}b2 {c2} c3 {c1} {sa}a1"
                self.assertEqual(self.rule.apply(self.get_basic_config(config), ("", 0, ""))[2],
                                 self.get_basic_config(expected) + "\n")

    def test_complex_inversion(self):
        configs = ["a1 <= b1", "a1 < b1", "a1 > b1", "a1 >= b1"]
        expecteds = ["b1 >= a1", "b1 > a1", "b1 < a1", "b1 <= a1"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_basic_config(c), ("", 0, ""))[2], self.get_basic_config(e)+"\n")

# ----------------------------------


class TestClearCondensedAssign(unittest.TestCase):
    """
    Test for the ClearCondensedAssign redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.ClearCondensedAssign("", False, None)

    def get_basic_config(self, config):
        """
        Function to get the configured reference code for the basic tests

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(a):
            i = 0
            i {config} a
            return i"""
        return code

    def test_basic(self):
        configs = ["+=", "-=", "/=", "*=", "&=", "|=", "^="]
        for c in configs:
            self.assertEqual(self.rule.apply(self.get_basic_config(c), ("", 0, ""))[2], self.get_basic_config("=")+"\n")

    def get_complex_config(self, config):
        """
        Function to get the configured reference code for the complex tests

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(a, b):
            if a >= b or b <= a:
                val = a == b
                def f(x, y, z, w, s = 0):
                    val[f0(val, [i for i in range(x+y*z)])] {config} x * y * z + s
                    return val == s
            return a"""
        return code

    def test_complex(self):
        configs = ["+=", "-=", "/=", "*=", "&=", "|=", "^="]
        for c in configs:
            self.assertEqual(self.rule.apply(self.get_complex_config(c), ("", 0, ""))[2], self.get_complex_config("=")+"\n")

# ----------------------------------


class TestUnravelCondensedAssign(unittest.TestCase):
    """
    Test for the UnravelCondensedAssign redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.UnravelCondensedAssign("", False, None)

    def get_basic_config(self, config):
        """
        Function to get the configured reference code for the basic tests

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(a):
            i = 0
            i {config} a
            return i"""
        return code

    def test_basic(self):
        configs = ["+=", "-=", "/=", "*=", "&=", "|=", "^="]
        expecteds = ["= i +", "= i -", "= i /", "= i *", "= i &", "= i |", "= i ^"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_basic_config(c), ("", 0, ""))[2],
                             self.get_basic_config(e) + "\n")

    def get_complex_config(self, config):
        """
        Function to get the configured reference code for the complex tests

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(a, b):
            if a >= b or b <= a:
                val = a == b
                def f(x, y, z, w, s = 0):
                    val {config} x * y * z + s
                    return val == s
            return a"""
        return code

    def test_complex(self):
        configs = ["+=", "-=", "/=", "*=", "&=", "|=", "^="]
        expecteds = ["= val +", "= val -", "= val /", "= val *", "= val &", "= val |", "= val ^"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_complex_config(c), ("", 0, ""))[2],
                             self.get_complex_config(e) + "\n")

    def get_complex_left_config(self, config):
        """
        Function to get the configured reference code for the complex tests

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(a, b):
            if a >= b or b <= a:
                val = a == b
                def f(x, y, z, w, s = 0):
                    val[f0(val, [i for i in range(x+y*z)])] {config} (x * y * z + s)
                    return val == s
            return a"""
        return code

    def test_complex_left(self):
        value = "val[f0(val, [i for i in range(x+y*z)])]"
        configs = ["+=", "-=", "/=", "*=", "&=", "|=", "^="]
        expecteds = [f"= {value} +", f"= {value} -", f"= {value} /", f"= {value} *", f"= {value} &", f"= {value} |", f"= {value} ^"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_complex_left_config(c), ("", 0, ""))[2],
                             self.get_complex_left_config(e) + "\n")

# ----------------------------------


class TestRavelAssign(unittest.TestCase):
    """
    Test for the RavelAssign redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RavelAssign("", False, None)

    def get_basic_config(self, config):
        """
        Function to get the configured reference code for the basic tests

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(a):
            i = 0
            i {config} a
            return i"""
        return code

    def test_basic(self):
        configs = ["= i +", "= i -", "= i /", "= i *", "= i &", "= i |", "= i ^"]
        expecteds = ["+=", "-=", "/=", "*=", "&=", "|=", "^="]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_basic_config(c), ("", 0, ""))[2],
                             self.get_basic_config(e) + "\n")

    def get_complex_config(self, config):
        """
        Function to get the configured reference code for the complex tests

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(a, b):
            if a >= b or b <= a:
                val = a == b
                def f(x, y, z, w, s = 0):
                    val {config} (x * y * z + s)
                    return val == s
            return a"""
        return code

    def test_complex(self):
        configs = ["= val +", "= val -", "= val /", "= val *", "= val &", "= val |", "= val ^"]
        expecteds = ["+=", "-=", "/=", "*=", "&=", "|=", "^="]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_complex_config(c), ("", 0, ""))[2],
                             self.get_complex_config(e) + "\n")

    def get_complex_left_config(self, config):
        """
        Function to get the configured reference code for the complex tests with a complex left-hand side

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(a, b):
            if a >= b or b <= a:
                val = a == b
                def f(x, y, z, w, s = 0):
                    val[f0(val, [i for i in range(x+y*z)])] {config} (x * y * z + s)
                    return val == s
            return a"""
        return code

    def test_complex_left(self):
        value = "val[f0(val, [i for i in range(x+y*z)])]"
        configs = [f"= {value} +", f"= {value} -", f"= {value} /", f"= {value} *", f"= {value} &", f"= {value} |", f"= {value} ^"]
        expecteds = ["+=", "-=", "/=", "*=", "&=", "|=", "^="]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_complex_left_config(c), ("", 0, ""))[2],
                             self.get_complex_left_config(e) + "\n")

# ----------------------------------


class TestAddReturnToInit(unittest.TestCase):
    """
    Test for the AddReturnToInit redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.AddReturnToInit("", False, None)

    def test_basic(self):
        code = """
        class Nice:
            def __init__(self):
                self.a = 0"""
        expected = """
        class Nice:
            def __init__(self):
                self.a = 0
                return self\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        class Complex(ABC):
            def other(self, init):
                class Other(Complex):
                    def init(init__):
                        return __init__
                    def __init__(self, _init_):
                        self.init = _init_
                        print(self.init())
                        super().__init__()
                        if self.a == True:
                            print(a)
                            while False:
                                self.init()
                return Other.__init__(init)
            def __init__(self):
                self.a = True"""
        expected = """
        class Complex(ABC):
            def other(self, init):
                class Other(Complex):
                    def init(init__):
                        return __init__
                    def __init__(self, _init_):
                        self.init = _init_
                        print(self.init())
                        super().__init__()
                        if self.a == True:
                            print(a)
                            while False:
                                self.init()
                        return self
                return Other.__init__(init)
            def __init__(self):
                self.a = True\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# ----------------------------------


class TestOutOfBoundRange(unittest.TestCase):
    """
    Test for the OutOfBoundRange redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.OutOfBoundRange("", False, None)

    def get_basic_config(self, config):
        """
        Function to get the configured reference code for the basic tests

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(n):
            for i in range({config}):
                print(i)
            return n"""
        return code

    def test_basic(self):
        configs = ["n", "2", "3.5", "1 + 2", "a + b", "a + 1", "1 + b", "len(n)"]
        expecteds = ["n + 1", "3", "4.5", "1 + 2 + 1", "a + b + 1", "a + 1 + 1", "1 + b + 1", "len(n) + 1"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_basic_config(c), ("", 0, ""))[2],
                             self.get_basic_config(e) + "\n")

    def test_integers(self):
        for i in range(0, 100):
            self.assertEqual(self.rule.apply(self.get_basic_config(i), ("", 0, ""))[2],
                             self.get_basic_config(i+1) + "\n")

    def test_negative_integers(self):
        for i in range(-100, 0):
            self.assertEqual(self.rule.apply(self.get_basic_config(i), ("", 0, ""))[2],
                             self.get_basic_config(str(i) + " + 1") + "\n")

    def test_floats(self):
        for i in range(0, 10):
            for v in range(0, 100):
                value = i + v/100
                self.assertEqual(self.rule.apply(self.get_basic_config(value), ("", 0, ""))[2],
                                 self.get_basic_config(value+1) + "\n")

    def test_negative_floats(self):
        for i in range(-10, 0):
            for v in range(0, 100):
                value = i + v/100
                self.assertEqual(self.rule.apply(self.get_basic_config(value), ("", 0, ""))[2],
                                 self.get_basic_config(str(value) + " + 1") + "\n")

    def test_expression(self):
        config = "(a + b)*c - len(w) + 10 // d ^ a"
        expected = "(a + b)*c - len(w) + 10 // d ^ a + 1"
        self.assertEqual(self.rule.apply(self.get_basic_config(config), ("", 0, ""))[2],
                         self.get_basic_config(expected) + "\n")

    def test_sequence(self):
        config = "[i if i < 10 else 0 for i in range(n) if i > 5]"
        expected = "[i if i < 10 else 0 for i in range(n) if i > 5] + 1"
        self.assertEqual(self.rule.apply(self.get_basic_config(config), ("", 0, ""))[2],
                         self.get_basic_config(expected) + "\n")

    def test_complex(self):
        code = """
        class Test:
            def fun(a, b, x):
                value[[i for i in np.arange(10)]] = (lambda x: a*x + b - math.ceil(math.floor(self.value)))(x)
                for i in range((lambda x: a*x + b - math.ceil(math.floor(self.value)))(x)):
                    value -= i
                return value"""
        expected = """
        class Test:
            def fun(a, b, x):
                value[[i for i in np.arange(10)]] = (lambda x: a*x + b - math.ceil(math.floor(self.value)))(x)
                for i in range((lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) + 1):
                    value -= i
                return value\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# ----------------------------------


class TestDivisionChange(unittest.TestCase):
    """
    Test for the DivisionChange redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.DivisionChange("", False, None)

    def get_basic_config(self, config):
        """
        Function to get the configured reference code for the basic tests

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(a, b):
            return a {config} b"""
        return code

    def test_basic(self):
        configs = ["/", "//"]
        expecteds = ["//", "/"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_basic_config(c), ("", 0, ""))[2],
                             self.get_basic_config(e) + "\n")

    def test_no_matches(self):
        configs = ["+", "*", "-", "%", "|", "&", "^", "@"]
        for c in configs:
            with self.assertRaises(NoMatches):
                self.rule.apply(self.get_basic_config(c), ("", 0, ""))

    def get_complex_config(self, config):
        """
        Function to get the configured reference code for the complex tests

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(a, b):
            if a >= b or b <= a:
                val /= a == b + "/x10\x20"
                def f(x, y, z, w, s = 0):
                    # Comment // is not allowed, nor /
                    val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])] = (x * y * z + s)
                    return val {config} s
            return a"""
        return code

    def test_complex(self):
        configs = ["/", "//"]
        expecteds = ["//", "/"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_complex_config(c), ("", 0, ""))[2],
                             self.get_complex_config(e) + "\n")

# ----------------------------------


class TestPrintBeforeReturn(unittest.TestCase):
    """
    Test for the PrintBeforeReturn redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.PrintBeforeReturn("", False, None)

    def test_basic(self):
        code = """
        def fun(n):
            return n"""
        expected = """
        def fun(n):
            print(n)
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun(n):
            try:
                def inner(n):
                    if True:
                        def in_inner(n):
                            class Inner:
                                def fct(n):
                                    print(a)
                                    print(b)
                                    if True:
                                        print(True)
                                        while True:
                                            print(a)
                                            print(b)
                                    return val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])] == (x * y * z + s)
                            return Inner
                        return n
                    return n
                return n
            except:
                pass"""
        expected = """
        def fun(n):
            try:
                def inner(n):
                    if True:
                        def in_inner(n):
                            class Inner:
                                def fct(n):
                                    print(a)
                                    print(b)
                                    if True:
                                        print(True)
                                        while True:
                                            print(a)
                                            print(b)
                                    print(val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])] == (x * y * z + s))
                                    return val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])] == (x * y * z + s)
                            return Inner
                        return n
                    return n
                return n
            except:
                pass\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# ----------------------------------


class TestReturnInIndentedBlock(unittest.TestCase):
    """
    Test for the ReturnInIndentedBlock redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.ReturnInIndentedBlock("", False, None)

    def get_basic_config(self, config, spaces):
        """
        Function to get the configured reference code for the basic tests

        Args:
            config: the patch to be applied on the reference code
            spaces: indentation level to add to the return

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(n):
            {config}:
                pass
            {spaces}return n"""
        return code

    def test_basic(self):
        configs = ["for i in range(n)", "if i < n", "while i < n", "with open(n, 'r') as file"]
        for c in configs:
            self.assertEqual(self.rule.apply(self.get_basic_config(c, ""), ("", 0, ""))[2],
                             self.get_basic_config(c, "    ")+"\n")

    def get_complex_config(self, c1, c2):
        """
        Function to get the configured reference code and the expected output for the complex tests

        Args:
            c1: the first configuration for the outer indented block
            c2: the second configuration for the second indented block

        Returns:
            the code and the expected output configured with the specific c1 and c2
        """
        code = f"""
        def fun(n):
            try:
                def inner(n):
                    if True:
                        def in_inner(n):
                            class Inner:
                                def fct(n):
                                    print(a)
                                    print(b)
                                    {c1}
                                        print(True)
                                        {c2}
                                            print(a)
                                            print(b)
                                    return val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])] == (x * y * z + s)
                            return Inner
                        return n
                    return n
                return n
            except:
                pass"""
        expected = f"""
        def fun(n):
            try:
                def inner(n):
                    if True:
                        def in_inner(n):
                            class Inner:
                                def fct(n):
                                    print(a)
                                    print(b)
                                    {c1}
                                        print(True)
                                        {c2}
                                            print(a)
                                            print(b)
                                        return val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])] == (x * y * z + s)
                            return Inner
                        return n
                    return n
                return n
            except:
                pass\n"""
        return code, expected

    def test_complex(self):
        configs = ["if value > 0:", "while value > 0:", "for i in range(n):", "with open(n, 'r') as file:",
                   "if value < 0:\n"+" "*40+"pass\n"+" "*36+"else:"]
        for c1 in configs:
            for c2 in configs[:-1]+["if value < 0:\n"+" "*44+"pass\n"+" "*40+"else:"]:
                code, expected = self.get_complex_config(c1, c2)
                self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_no_matches_1(self):
        code = """
        def fun(n):
            return n"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_2(self):
        code = """
        def fun(n):
            def in(n):
                pass
            return n"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_3(self):
        code = """
        def fun(n):
            try:
                pass
            except:
                print("error")
            return n"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_4(self):
        code = """
        def fun(n):
            {block}:
                print(True)
            print(False)
            return n"""
        configs = ["for i in range(n)", "if i < n", "while i < n", "with open(n, 'r') as file"]
        for c in configs:
            with self.assertRaises(NoMatches):
                self.rule.apply(code.format(block=c), ("", 0, ""))

# ----------------------------------


class TestBadOpenMode(unittest.TestCase):
    """
    Test for the BadOpenMode redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.BadOpenMode("", False, None)

    def get_basic_config(self, config):
        """
        Function to get the configured reference code for the basic tests

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(name):
            with open(name, '{config}') as file:
                pass
            return name"""
        return code

    def test_basic(self):
        configs = ["r", "w"]
        expecteds = ["w", "r"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_basic_config(c), ("", 0, ""))[2],
                             self.get_basic_config(e) + "\n")

    def get_complex_config(self, config):
        """
        Function to get the configured reference code for the complex tests

        Args:
            config: the patch to be applied on the reference code

        Returns:
            the code configured with the specific config
        """
        code = f"""
        def fun(n):
            try:
                def inner(n):
                    with torch.no_grads():
                        def in_inner(n):
                            class Inner:
                                def fct(n):
                                    print(a)
                                    print(b)
                                    with torch.device(cpu) as d, ressource as r:
                                        print(True)
                                        with val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])] == (x * y * z + s), open(file, '{config}') as f:
                                            print(a)
                                            print(b)
                                    return r
                            return Inner
                        return n
                    return n
                return n
            except:
                pass"""
        return code

    def test_complex(self):
        configs = ["r", "w"]
        expecteds = ["w", "r"]
        for c, e in zip(configs, expecteds):
            self.assertEqual(self.rule.apply(self.get_complex_config(c), ("", 0, ""))[2],
                             self.get_complex_config(e)+"\n")

    def test_in_call(self):
        code = """
        def fun(fct, f):
            fct(open(f, 'r'))"""
        expected = """
        def fun(fct, f):
            fct(open(f, 'w'))\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_in_key(self):
        code = """
        def fun(d, f):
            return d[open(f, 'r')]"""
        expected = """
        def fun(d, f):
            return d[open(f, 'w')]\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_in_slice(self):
        code = """
        def fun(l, f):
            return l[:open(f, 'r')]"""
        expected = """
        def fun(l, f):
            return l[:open(f, 'w')]\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_classic_call(self):
        code = """
        def fun(f):
            file = open(f, 'r')
            return file.read()"""
        expected = """
        def fun(f):
            file = open(f, 'w')
            return file.read()\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_chained_call(self):
        code = """
        def fun(f):
            return open(f, 'r').read()"""
        expected = """
        def fun(f):
            return open(f, 'w').read()\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_a_match_but_not_first(self):
        code = """
        def fun(f):
            with open(f, mode) as file:
                pass
            with open(f, 'r') as file:
                pass"""
        expected = """
        def fun(f):
            with open(f, mode) as file:
                pass
            with open(f, 'w') as file:
                pass\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_no_matches_lambda(self):
        code = """
        def fun(n):
            return (lambda x: open)(x)"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_variable(self):
        code = """
        def fun(open):
            # Should print open
            open = open + 1
            print(open)
            return open"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_def(self):
        code = """
        def open(path, mode):
            return path"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_chain(self):
        code = """
        def fun(f):
            return f.open(path, 'r')"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_mode(self):
        code = """
        def fun(f):
            file = open(f, mode)
            return file"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

# ----------------------------------


class TestNoClose(unittest.TestCase):
    """
    Test for the NoClose redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.NoClose("", False, None)

    def test_basic(self):
        code = """
        def fun(path):
            file = open(path, 'r')
            file.close()"""
        expected = """
        def fun(path):
            file = open(path, 'r')\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_after_close(self):
        code = """
        def fun(path):
            file = open(path, 'r')
            content = file.read()
            file.close()
            print(content)"""
        expected = """
        def fun(path):
            file = open(path, 'r')
            content = file.read()
            print(content)\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun(n):
            try:
                def inner(n):
                    with torch.no_grads():
                        def in_inner(n):
                            class Inner:
                                def fct(n):
                                    print(a)
                                    print(b)
                                    with torch.device(cpu) as d, ressource as r:
                                        print(True)
                                        val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])] = open(file, 'r')
                                        print(a)
                                        print(b)
                                    return r
                            return Inner
                        return n
                    return n
                if True:
                    print(True)
                val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])].close()
                while True:
                    print(n)
                return n
            except:
                pass"""

        expected = """
        def fun(n):
            try:
                def inner(n):
                    with torch.no_grads():
                        def in_inner(n):
                            class Inner:
                                def fct(n):
                                    print(a)
                                    print(b)
                                    with torch.device(cpu) as d, ressource as r:
                                        print(True)
                                        val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])] = open(file, 'r')
                                        print(a)
                                        print(b)
                                    return r
                            return Inner
                        return n
                    return n
                if True:
                    print(True)
                while True:
                    print(n)
                return n
            except:
                pass\n"""

        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_no_matches_in_call(self):
        code = """
        def fun(fct, f):
            file = open(f, 'r')
            fct(file.close())"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_in_key(self):
        code = """
        def fun(d, f):
            file = open(f, 'r')
            return d[file.close()]"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_in_slice(self):
        code = """
        def fun(l, f):
            file = open(f, 'r')
            return l[:file.close()]"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_chained_call(self):
        code = """
        def fun(f):
            return open(f, 'r').read().close()"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_lambda(self):
        code = """
        def fun(n):
            return (lambda x: close)(x)"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_variable(self):
        code = """
        def fun(close):
            # Should print close
            close = close + 1
            print(close)
            return close"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_function(self):
        code = """
        def close(path, mode):
            return path"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_not_attribute(self):
        code = """
        def fun(f):
            close()"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

# ----------------------------------


class TestMissExcept(unittest.TestCase):
    """
    Test for the MissExcept redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.MissExcept("", False, None)

    def test_basic(self):
        code = """
        def fun(n):
            try:
                n = n / 0
            except:
                print(n)"""
        expected = """
        def fun(n):
            try:
                n = n / 0
            """
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_after_except(self):
        code = """
        def fun(n):
            try:
                n = n / 0
            except:
                print(n)
            return n"""
        expected = """
        def fun(n):
            try:
                n = n / 0
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_if_before_return_after_except(self):
        code = """
        def fun(n):
            try:
                n = n / 0
                if True:
                    print(True)
            except:
                print(n)
            return n"""
        expected = """
        def fun(n):
            try:
                n = n / 0
                if True:
                    print(True)
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_multiple_excepts(self):
        code = """
        def fun(n):
            try:
                n = n / 0
                if True:
                    print(True)
            except DivisionError as d:
                print(d)
            except Exception as e:
                print(e)
            return n"""
        expected = """
        def fun(n):
            try:
                n = n / 0
                if True:
                    print(True)
            except Exception as e:
                print(e)
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun(n):
            def f(a, b, c):
                try:
                    class ABC:
                        def __init__(self):
                            try:
                                self.a = a * b / c
                            finally:
                                self.a = 0
                        def _try(self):
                            # Should not try
                            pass
                        def _except(self):
                            # Should except
                            self._except = 0
                except Exception as exception:
                    try:
                        raise ValueError(0)
                    except ValueError as ve:
                        print(ve)
                    except Exception as _:
                        print('_')
                finally:
                    return ABC
            return f"""
        expected = """
        def fun(n):
            def f(a, b, c):
                try:
                    class ABC:
                        def __init__(self):
                            try:
                                self.a = a * b / c
                            finally:
                                self.a = 0
                        def _try(self):
                            # Should not try
                            pass
                        def _except(self):
                            # Should except
                            self._except = 0
                finally:
                    return ABC
            return f\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_no_matches_normal_code(self):
        code = """
        def fun(n):
            return n"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_try(self):
        code = """
        def fun(n):
            try:
                return n"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

    def test_no_matches_try_finally(self):
        code = """
        def fun(n):
            try:
                return n
            finally:
                return 0"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

# ----------------------------------


class TestReplaceAllOccurrenceOfAVariable(unittest.TestCase):
    """
    Test for the ReplaceAllOccurrenceOfAVariable redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.ReplaceAllOccurrenceOfAVariable("", False, None)
        self.no_default_rule = rr.ReplaceAllOccurrenceOfAVariable("", False, None, "replace_all_var r")

    def test_basic(self):
        code = """
        def fun(n):
            return n"""
        expected = """
        def fun(a):
            return a\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)
        self.assertEqual(self.no_default_rule.apply(code, ("", 0, ""))[2], expected)

    def test_name_attribute(self):
        code = """
        def fun(append):
            l = [a, b, c]
            l.append(append)
            return append, l"""
        expected = """
        def fun(x):
            l = [a, b, c]
            l.append(x)
            return x, l\n"""
        expected_no_default = """
        def fun(d):
            l = [a, b, c]
            l.append(d)
            return d, l\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)
        self.assertEqual(self.no_default_rule.apply(code, ("", 0, ""))[2], expected_no_default)

    def test_recursive(self):
        code = """
        def fun(v):
            print(a, b, c, x, y, z, w, var, variable)
            if v == 0:
                return 1
            return fun(v-1)*v"""
        expected = """
        def fun(d):
            print(a, b, c, x, y, z, w, var, variable)
            if d == 0:
                return 1
            return fun(d-1)*d\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)
        self.assertEqual(self.no_default_rule.apply(code, ("", 0, ""))[2], expected)

    def test_recursive_no_arg(self):
        code = """
        def fun():
            return fun"""
        expected = """
        def a():
            return a\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)
        self.assertEqual(self.no_default_rule.apply(code, ("", 0, ""))[2], expected)

    def test_function_call(self):
        code = """
        def fun(f, a, b, c):
            return f(a * b + c)"""
        expected = """
        def fun(x, a, b, c):
            return x(a * b + c)\n"""
        expected_no_default = """
        def fun(d, a, b, c):
            return d(a * b + c)\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)
        self.assertEqual(self.no_default_rule.apply(code, ("", 0, ""))[2], expected_no_default)

    def test_ignore_string(self):
        code = """
        def fun():
            print('a')
            return a"""
        expected = """
        def fun():
            print('a')
            return b\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)
        self.assertEqual(self.no_default_rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex_lambda(self):
        code = """
        def fun():
            return ([lambda x: a*x + b - math.ceil(math.floor(self.value))(x) for i in range(x+y*z)])"""
        expected = """
        def fun():
            return ([lambda c: a*c + b - math.ceil(math.floor(self.value))(c) for i in range(c+y*z)])\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)
        self.assertEqual(self.no_default_rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun():
            def f():
                try:
                    class ABC:
                        def __init__(self):
                            try:
                                self.a = a * b / c
                            finally:
                                self.a = 0
                        def _try(self):
                            # Should not try
                            pass
                        def _except(self):
                            # Should except
                            self._except = 0
                except Exception as exception:
                    try:
                        test.self.value = 0
                        raise ValueError(0)
                    except ValueError as ve:
                        print(ve)
                    except Exception as _:
                        print('_')
                finally:
                    return ABC
            return f"""
        expected = """
        def fun():
            def f():
                try:
                    class ABC:
                        def __init__(x):
                            try:
                                x.a = a * b / c
                            finally:
                                x.a = 0
                        def _try(x):
                            # Should not try
                            pass
                        def _except(x):
                            # Should except
                            x._except = 0
                except Exception as exception:
                    try:
                        test.self.value = 0
                        raise ValueError(0)
                    except ValueError as ve:
                        print(ve)
                    except Exception as _:
                        print('_')
                finally:
                    return ABC
            return f\n"""
        expected_no_default = """
        def fun():
            def f():
                try:
                    class ABC:
                        def __init__(d):
                            try:
                                d.a = a * b / c
                            finally:
                                d.a = 0
                        def _try(d):
                            # Should not try
                            pass
                        def _except(d):
                            # Should except
                            d._except = 0
                except Exception as exception:
                    try:
                        test.self.value = 0
                        raise ValueError(0)
                    except ValueError as ve:
                        print(ve)
                    except Exception as _:
                        print('_')
                finally:
                    return ABC
            return f\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)
        self.assertEqual(self.no_default_rule.apply(code, ("", 0, ""))[2], expected_no_default)

    def test_no_matches_all_used(self):
        code = """
        def fun(a, b, c):
            x, y, z, w = (0, 0, 0, 0)
            var = variable
            others = (d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v)
            return a * b + c"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))
        with self.assertRaises(NoMatches):
            self.no_default_rule.apply(code, ("", 0, ""))

    def test_no_matches_all_builtin(self):
        code = """
        def fun():
            # This code does not make sense bu this is intentional
            abs = all(any(ascii()))
            bin().bool().bytearray().bytes()
            breakpoint()
            complex = callable(), chr(), classmethod(), compile()
            delattr(dict(dir(divmod())))
            for eval in enumerate(exec):
                if filter or float and format:
                    while frozenset:
                        getattr(globals)
                    with hasattr(hash, help) as hex:
                        from id.input import int as isinstance
                        iter(issubclass)
            map = min([max(len(list(locals())))])
            memoryview()
            next()
            ord.open(object(), oct())
            print(pow, property)
            if True ^ False & None | 0:
                reversed(range(round(repr())))
            elif True:
                set(setattr())
            else:
                slice()
            sorted()
            str.staticmethod()
            super().sum()
            print(type(tuple()))
            zip(vars(), vars())
            raise False
            return not (lambda __import__: 0) is None"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))
        with self.assertRaises(NoMatches):
            self.no_default_rule.apply(code, ("", 0, ""))

# ----------------------------------


class TestRenameAllDef(unittest.TestCase):
    """
    Test for the RenameAllDef redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RenameAllDef("", False, None)

    def test_basic(self):
        code = """
        def fun(n):
            return n"""
        expected = """
        def function_1(n):
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_two_functions(self):
        code = """
        def fun(n):
            return n
        def run(n):
            return n"""
        expected = """
        def function_1(n):
            return n
        def function_2(n):
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_redef_functions(self):
        code = """
        def fun(n):
            return n
        def fun(n):
            return n"""
        expected = """
        def function_1(n):
            return n
        def function_1(n):
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_redef_functions_used(self):
        code = """
        def fun(n):
            return fun(n-1) * n
        def fun(n):
            return fun(n-1) * n"""
        expected = """
        def function_1(n):
            return function_1(n-1) * n
        def function_1(n):
            return function_1(n-1) * n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_variable(self):
        code = """
        fun = 0
        def fun(n):
            fun = 0
            return n"""
        expected = """
        function_1 = 0
        def function_1(n):
            function_1 = 0
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_recursive(self):
        code = """
        def fun(n):
            if n == 0:
                return 1
            return fun(n-1) * n"""
        expected = """
        def function_1(n):
            if n == 0:
                return 1
            return function_1(n-1) * n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_nested_def(self):
        code = """
        def fun(n):
            def run(m):
                return m * n, fun(m)
            return run"""
        expected = """
        def function_1(n):
            def function_2(m):
                return m * n, function_1(m)
            return function_2\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun(n):
            try:
                def inner(n):
                    with torch.no_grads():
                        def in_inner(n):
                            class Inner:
                                def fct(n):
                                    print(a)
                                    print(b)
                                    with torch.device(cpu) as d, ressource as r:
                                        print(True)
                                        val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])] = open(file, 'r')
                                        print(a)
                                        print(b)
                                    return fun, inner, in_inner, fct
                            return Inner
                        return n
                    return n
                if True:
                    print(True)
                val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])].close()
                while True:
                    print(n)
                return n
            except:
                pass"""
        expected = """
        def function_1(n):
            try:
                def function_2(n):
                    with torch.no_grads():
                        def function_3(n):
                            class Inner:
                                def function_4(n):
                                    print(a)
                                    print(b)
                                    with torch.device(cpu) as d, ressource as r:
                                        print(True)
                                        val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])] = open(file, 'r')
                                        print(a)
                                        print(b)
                                    return function_1, function_2, function_3, function_4
                            return Inner
                        return n
                    return n
                if True:
                    print(True)
                val[f0(val, [(lambda x: a*x + b - math.ceil(math.floor(self.value)))(x) for i in range(x+y*z)])].close()
                while True:
                    print(n)
                return n
            except:
                pass\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# ----------------------------------


class TestRenameAllVarsDummy(unittest.TestCase):
    """
    Test for the RenameAllVarsDummy redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RenameAllVarsDummy("", False, None)

    def test_basic(self):
        code = """
        def fun(a, b):
            return a * b"""
        expected = """
        def fun(var_1, var_2):
            return var_1 * var_2\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_redef(self):
        code = """
        def fun(a, b):
            a = b
            b = a
            return a * b"""
        expected = """
        def fun(var_1, var_2):
            var_1 = var_2
            var_2 = var_1
            return var_1 * var_2\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_variable(self):
        code = """
        fun = 0
        def fun(n):
            fun = 0
            return n"""
        expected = """
        var_1 = 0
        def var_1(var_2):
            var_1 = 0
            return var_2\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_recursive(self):
        code = """
        def fun(n):
            if n == 0:
                return 1
            return fun(n-1) * n"""
        expected = """
        def var_2(var_1):
            if var_1 == 0:
                return 1
            return var_2(var_1-1) * var_1\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_already_used(self):
        code = """
        def var_1(n):
            if n == 0:
                return 1
            return var_1(n-1) * n"""
        expected = """
        def var_2(var_1):
            if var_1 == 0:
                return 1
            return var_2(var_1-1) * var_1\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun(i):
            try:
                def inner(i):
                    with torch.no_grads():
                        def in_inner(i):
                            class Inner:
                                def fct(i):
                                    print(a)
                                    print(b)
                                    with torch.device(cpu) as d, ressource as r:
                                        print(True)
                                        val[f0(val, [(lambda x: a*x + b)(x) for i in range(x)])] = 0
                                        print(a)
                                        print(b)
                                    return i
                            return Inner
                        return i
                    return i
                if True:
                    print(True)
                val[f0(val, [(lambda x: a*x + b)(x) for i in range(x)])] = 0
                while True:
                    print(i)
                return i
            except:
                pass"""
        expected = """
        def fun(var_1):
            try:
                def inner(var_1):
                    with var_2.no_grads():
                        def in_inner(var_1):
                            class Inner:
                                def fct(var_1):
                                    print(var_3)
                                    print(var_4)
                                    with var_2.device(var_5) as var_6, var_7 as var_8:
                                        print(True)
                                        var_9[var_10(var_9, [(lambda var_11: var_3*var_11 + var_4)(var_11) for var_1 in range(var_11)])] = 0
                                        print(var_3)
                                        print(var_4)
                                    return var_1
                            return var_12
                        return var_1
                    return var_1
                if True:
                    print(True)
                var_9[var_10(var_9, [(lambda var_11: var_3*var_11 + var_4)(var_11) for var_1 in range(var_11)])] = 0
                while True:
                    print(var_1)
                return var_1
            except:
                pass\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_identity_all_builtin(self):
        code = """
        def fun():
            # This code does not make sense bu this is intentional
            abs = all(any(ascii()))
            bin().bool().bytearray().bytes()
            breakpoint()
            complex = callable(), chr(), classmethod(), compile()
            delattr(dict(dir(divmod())))
            for eval in enumerate(exec):
                if filter or float and format:
                    while frozenset:
                        getattr(globals)
                    with hasattr(hash, help) as hex:
                        from id.input import int as isinstance
                        iter(issubclass)
            map = min([max(len(list(locals())))])
            memoryview()
            next()
            ord.open(object(), oct())
            print(pow, property)
            if True ^ False & None | 0:
                reversed(range(round(repr())))
            elif True:
                set(setattr())
            else:
                slice()
            sorted()
            str.staticmethod()
            super().sum()
            print(type(tuple()))
            zip(vars(), vars())
            raise False
            return not (lambda __import__: 0) is None"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], code+"\n")

# ----------------------------------


class TestRemoveCommentsAndDocstrings(unittest.TestCase):
    """
    Test for the RemoveCommentsAndDocstrings redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RemoveCommentsAndDocstrings("", False, None)

    def test_basic(self):
        code = """
        #comment
        print(True)"""
        expected = """
        print(True)\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_comments_in_function(self):
        code = """
        # Should always comment
        def fun(n):
            # Return n
            return n"""
        expected = """
        def fun(n):
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_comments_after_return(self):
        code = """
        # Should always comment
        def fun(n):
            # Return n
            return n
            # This returned n"""
        expected = """
        def fun(n):
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_comments_after_if(self):
        code = """
        # Should always comment
        def fun(n):
            if False:
                # Do nothing
                pass
            # Return n
            return n"""
        expected = """
        def fun(n):
            if False:
                pass
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_comments_last_in_if(self):
        code = """
        # Should always comment
        def fun(n):
            if False:
                pass
                # Do nothing
            return n"""
        expected = """
        def fun(n):
            if False:
                pass
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_comments_alone_in_if(self):
        code = """
        # Should always comment
        def fun(n):
            if False:
                # Do nothing
            return n"""
        expected = """
        def fun(n):
            if False:
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_comments_end_line_in_if(self):
        code = """
        # Should always comment
        def fun(n):
            if False:
                pass # Do nothing
            return n"""
        expected = """
        def fun(n):
            if False:
                pass
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_comments_end_line(self):
        code = """
        def fun(n):  # A nice function
            if False:  # Is never True
                pass  # Does nothing
            return n  # Returns n"""
        expected = """
        def fun(n):  
            if False:  
                pass
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_docstrings_in_function(self):
        code = """
        # Should always comment
        def fun(n):
            '''
            Function fun
            '''
            # Return n
            return n"""
        expected = """
        def fun(n):
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_classic_string_vs_docstring(self):
        code = """
        # Should always comment
        def fun(n):
            '''
            Function fun
            '''
            # Return n
            return n + 'hello'"""
        expected = """
        def fun(n):
            return n + 'hello'\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_double_docstring(self):
        code = """
        # Should always comment
        def fun(n):
            '''
            Function fun 1
            '''
            # Return n
            '''
            Function fun 2
            '''
            return n"""
        expected = """
        def fun(n):
            '''
            Function fun 2
            '''
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_double_docstring_2(self):
        code = """
        # Should always comment
        def fun(n):
            # This is the docstring
            '''
            Function fun 1
            '''
            # Return n
            '''
            Function fun 2
            '''
            return n"""
        expected = """
        def fun(n):
            '''
            Function fun 2
            '''
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_double_functions(self):
        code = """
        # Should always comment
        def fun(n):
            # This is the docstring
            '''
            Function fun 1
            '''
            pass
        def fun_2(n):
            # Return n
            '''
            Function fun 2
            '''
            return n"""
        expected = """
        def fun(n):
            pass
        def fun_2(n):
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

# ----------------------------------


class TestRemoveParenthesis(unittest.TestCase):
    """
    Test for the RemoveParenthesis redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.RemoveParenthesis("", False, None)

    def test_basic(self):
        code = """
        def fun(a, b, c):
            return (a + b) * c"""
        expected = """
        def fun(a, b, c):
            return a + b * c\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun(i):
            try:
                def inner(i):
                    with torch.no_grads():
                        def in_inner(i):
                            class Inner:
                                def fct(i):
                                    print(a)
                                    print(b)
                                    with torch.device(cpu) as d, ressource as r:
                                        print(True)
                                        val[f0(val, [(lambda x: (a*x) + b)(x) for i in range(x)])] = (0, 0)
                                        print(a)
                                        print(b)
                                    return (i, val)
                            return Inner
                        return i
                    return i
                if True:
                    print(True)
                val[f0(val, [(lambda x: (a*x) + b)(x) for i in range(x)])] = 0
                while True:
                    print(i)
                return i
            except:
                pass"""
        expected = """
        def fun(i):
            try:
                def inner(i):
                    with torch.no_grads():
                        def in_inner(i):
                            class Inner:
                                def fct(i):
                                    print(a)
                                    print(b)
                                    with torch.device(cpu) as d, ressource as r:
                                        print(True)
                                        val[f0(val, [lambda x: (a*x) + b(x) for i in range(x)])] = (0, 0)
                                        print(a)
                                        print(b)
                                    return (i, val)
                            return Inner
                        return i
                    return i
                if True:
                    print(True)
                val[f0(val, [(lambda x: (a*x) + b)(x) for i in range(x)])] = 0
                while True:
                    print(i)
                return i
            except:
                pass\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_no_matches(self):
        code = """
        def fun(f, n):
            print(f, n)
            if False:
                fun(f, n)
            for i in range(n):
                a, b = f(i)
                x = (a, b)
            return x"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))

# ----------------------------------


class TestHardcodeArg(unittest.TestCase):
    """
    Test for the HardcodeArg redbaron mutation rule
    """
    def setUp(self) -> None:
        self.rule = rr.HardcodeArg("", False, None)

    def test_basic(self):
        code = """
        def fun(n):
            return n"""
        expected = """
        def fun(n):
            n = 0
            return n\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_complex(self):
        code = """
        def fun(i = 0):
            try:
                def inner(i = 0):
                    with torch.no_grads():
                        def in_inner(i = 0):
                            class Inner:
                                def fct(self, i = 0, j = 0, k = 0, another_variable_which_is_very_long):
                                    print(a)
                                    print(b)
                                    with torch.device(cpu) as d, ressource as r:
                                        print(True)
                                        val[f0(val, [(lambda x: (a*x) + b)(x) for i in range(x)])] = (0, 0)
                                        print(a)
                                        print(b)
                                    return (i, val)
                            return Inner
                        return i
                    return i
                if True:
                    print(True)
                val[f0(val, [(lambda x: (a*x) + b)(x) for i in range(x)])] = 0
                while True:
                    print(i)
                return i
            except:
                pass"""
        expected = """
        def fun(i = 0):
            try:
                def inner(i = 0):
                    with torch.no_grads():
                        def in_inner(i = 0):
                            class Inner:
                                def fct(self, i = 0, j = 0, k = 0, another_variable_which_is_very_long):
                                    another_variable_which_is_very_long = 0
                                    print(a)
                                    print(b)
                                    with torch.device(cpu) as d, ressource as r:
                                        print(True)
                                        val[f0(val, [(lambda x: (a*x) + b)(x) for i in range(x)])] = (0, 0)
                                        print(a)
                                        print(b)
                                    return (i, val)
                            return Inner
                        return i
                    return i
                if True:
                    print(True)
                val[f0(val, [(lambda x: (a*x) + b)(x) for i in range(x)])] = 0
                while True:
                    print(i)
                return i
            except:
                pass\n"""
        self.assertEqual(self.rule.apply(code, ("", 0, ""))[2], expected)

    def test_no_matches(self):
        code = """
        def fun(i = 0):
            try:
                def inner(i = 0):
                    with torch.no_grads():
                        def in_inner(i = 0):
                            class Inner:
                                def fct(self, i = 0, j = 0, k = 0):
                                    print(a)
                                    print(b)
                                    with torch.device(cpu) as d, ressource as r:
                                        print(True)
                                        val[f0(val, [(lambda x: (a*x) + b)(x) for i in range(x)])] = (0, 0)
                                        print(a)
                                        print(b)
                                    return (i, val)
                            return Inner
                        return i
                    return i
                if True:
                    print(True)
                val[f0(val, [(lambda x: (a*x) + b)(x) for i in range(x)])] = 0
                while True:
                    print(i)
                return i
            except:
                pass"""
        with self.assertRaises(NoMatches):
            self.rule.apply(code, ("", 0, ""))


if __name__ == '__main__':
    unittest.main()
