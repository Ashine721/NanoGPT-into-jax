import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

for folder in ['config', 'data', 'model', 'training', 'inference', 'utils']:
    path = os.path.join(BASE_DIR, folder, '__init__.py')
    if not os.path.exists(path):
        open(path, 'w').close()
        print(f'建立 {path}')
    else:
        print(f'已存在 {path}')