import sys
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class MHLogger:
    def __init__(self, file_like_obj):
        self.file_like_obj = file_like_obj

    def __call__(self, str_, log=True, next_line=False, stdout=False):
        my_str = f"{str_}"
        if next_line:
            my_str = my_str + "\n"
        if log:
            self.file_like_obj.write(my_str)
        if stdout:
            print(my_str, end='')

    def flush(self):
        sys.stdout.flush()
        self.file_like_obj.flush()
