from setuptools import setup

setup(name='transglot',
      version='0.1',
      description='Transglot: Learning Learning Local Parts of 3D Structure from Language.',
      url='http://github.com/63days/transglot',
      author='Panos Achlioptas, Juil Koo',
      author_email='optas@cs.stanford.edu, 63days@kaist.ac.kr',
      license='MIT',
      packages=['transglot'],
      install_requires=['pandas', 'torch', 'Pillow', 'numpy', 'matplotlib', 'six', 'nltk',
                        'pytorch-lightning==1.2.8', 'hydra-core==1.0.6', 'omegaconf==2.0.6'],
      zip_safe=False)
