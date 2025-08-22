#just a set of utils functions for crmstudio package
import re

def snake_to_pascal(snake_str):
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)

def pascal_to_snake(pascal_str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', pascal_str)
    class_snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return class_snake        