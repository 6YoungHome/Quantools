

class MissArgsError(Exception):
    def __init__(self, arg_name, message):
        self.arg_name = arg_name
        self.message = message

    def __str__(self):
        return f"缺失必要的参数{self.arg_name}\n{self.message}"


class InvalidArgsError(Exception):
    def __init__(self, arg_name, arg_value, message):
        self.arg_name = arg_name
        self.arg_value = arg_value
        self.message = message

    def __str__(self):
        return f"参数{self.arg_name}得到的错误的值{self.arg_value}\n{self.message}"