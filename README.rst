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

Introduction
============

This project allows you to leverage the output of Pupil Cloud's `Face Mapper <https://docs.pupil-labs.com/neon/pupil-cloud/enrichments/face-mapper/>`__ and map gaze on facial landmarks detected in the scene camera.
It generates a new visualization with the detected facial landmarks overlaid on the video and gaze on top, and also generates a new CSV file with the mapped gaze on facial landmarks. 

Requirements
============
You should have Python 3.9 or higher.

Installation
============

Make sure you have tkinter module installed, this usually comes with most modern python distributions, but it might be different for you if for example, you installed python using homebrew.

Check if you have tkinter installed: 

::

    python -m tkinter

If you don't have it installed, install it:

::

    brew install python-tk@3.11 # Change according to your python version

In order to download the package, you can simply run the following command from the terminal:

::

   git clone https://github.com/pupil-labs/gaze-on-facial-landmarks.git

Optional, but highly recommended: Create a virtual environment!

::

      python3.11 -m venv venv
      source venv/bin/activate

Go to the folder directory and install the dependencies

::

   cd your_directory/gaze-on-facial-landmarks
   pip install -e . 

Run it!
========

Run it and map gaze on facial landmarks!

::

   pl-gaze-on-facial-landmarks  

Once you hit run, a pop-up window will appear prompting you to select the following:

- Face Mapper output folder: The enrichment folder downloaded from Pupil Cloud

- Raw data output folder: The "Timeseries Data + Scene Video" downloaded from Pupil Cloud

- AOI radius: The radius of the circle to be used as an AOI mask for the eyes and nose (default: 30)

- Ellipse size: The size of the ellipse to be used as an AOI mask for the mouth (default: 30)

- Gaze circle size: The size of the usual red circle that reflects the gaze points (default: 20)

Support
========

For any questions/bugs, reach out to our `Discord server <https://pupil-labs.com/chat/>`__  or send us an email to info@pupil-labs.com. 
