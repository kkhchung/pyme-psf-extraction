#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.command.develop import develop

class run_post_develop(develop):
    def run(self):
        develop.run(self)
        
        from psf_extraction.plugins import install_plugin
        install_plugin.main()
        

setup(name='psf_extraction',
      version='0.1',
      description='psf extraction',
      author='Kenny Chung',
      author_email='kenny.chung@yale.edu',
      url='',
      packages=find_packages(),
      # package_data={
      #       # include all svg and html files, otherwise conda will miss them
      #       '': ['*.svg', '*.html'],
      # }
      cmdclass = {
              'develop': run_post_develop,
              },
     )
