.. image:: https://img.shields.io/pypi/v/skeleton.svg
   :target: `PyPI link`_

.. image:: https://img.shields.io/pypi/pyversions/skeleton.svg
   :target: `PyPI link`_

.. _PyPI link: https://pypi.org/project/skeleton

.. image:: https://github.com/jaraco/skeleton/workflows/tests/badge.svg
   :target: https://github.com/jaraco/skeleton/actions?query=workflow%3A%22tests%22
   :alt: tests

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: Black

.. .. image:: https://readthedocs.org/projects/skeleton/badge/?version=latest
..    :target: https://skeleton.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/skeleton-2022-informational
   :target: https://blog.jaraco.com/skeleton

TBD
============
- Color map fixated landmark 
- Check cloud fix on data output
- Check that everything works with other events 

Introduction
============

This project allows you to leverage the output of Pupil Cloud's `Face Mapper <https://docs.pupil-labs.com/neon/pupil-cloud/enrichments/face-mapper/>`__ and map fixations on facial landmarks detected in the scene camera.
It generates a new visualization with the detected facial landmarks overlaid on the video and fixations/gaze on top, and also generates a new CSV file with the mapped fixations on facial landmarks. 

Requirements
============
You should have Python 3.9 or higher.

Installation
============

Make sure you have tkinter module installed, this usually comes with most modern python distributions, but it might be different for you if for example, you installed python using homebrew.

Check if you have tkinter installed: 

..  code-block:: python

    python -m tkinter

If you don't have it installed, install it:

..  code-block:: python

    brew install python-tk@3.11 # Change according to your python version

In order to download the package, you can simply run the following command from the terminal:

..  code-block:: python
   git clone https://github.com/pupil-labs/fixations-on-face.git

Optional, but highly recommended: Create a virtual environment!

..  code-block:: python    
      python3.11 -m venv venv
      source venv/bin/activate

Go to the folder directory and install the dependencies

.. code-block:: python
   cd your_directory/map_fixations_on_face
   pip install -e . 

Run it!
========
.. code-block:: python
   pl-fixations-on-face

For any questions/bugs, reach out to our `Discord server<https://pupil-labs.com/chat/>` or send us an email to info@pupil-labs.com. 