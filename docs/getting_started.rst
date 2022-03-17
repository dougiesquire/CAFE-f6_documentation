Getting started
===============

Key steps for getting set up are handled using `make <https://www.gnu.org/software/make/>`_:

#. ``make environment`` creates the python environment or updates it if it exists
#. ``make data`` prepares the raw data (in ``data/raw``) for subsequent analysis. The processed data are stored in ``data/processed``. See :ref:`Data Preparation`
#. ``make skill`` calculates the skill metrics from the processed data. See :ref:`Calculating skill metrics`
#. ``make docs`` rebuilds this documentation
#. ``make clean`` cleans up unneeded files and directories
#. ``make lint`` runs ``black`` and ``flake8`` on ``src``
