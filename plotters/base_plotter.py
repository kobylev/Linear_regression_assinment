"""
Base Plotter Module
Contains the base class for all plotters.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from config import setup_logging

# Setup logger
logger = setup_logging(__name__)

class BasePlotter:
    """
    Base class for all plotters with common functionalities.
    """
    
    def __init__(self, figsize=(15, 12), style='whitegrid'):
        """Initializes the plotter with font fixes.

        Args:
            figsize (tuple, optional): Default figure size. Defaults to (15, 12).
            style (str, optional): Seaborn style. Defaults to 'whitegrid'.
        """
        self.figsize = figsize
        
        # Set matplotlib to use fonts that support standard characters
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False  # Use ASCII minus sign
        
        plt.style.use('default')
        sns.set_style(style)
        sns.set_palette("husl")
        logger.info("Initialized BasePlotter with default styles.")

    def _create_figure(self, num_subplots=1, **kwargs):
        """Creates a figure and axes.

        Args:
            num_subplots (int, optional): The number of subplots. Defaults to 1.
            **kwargs: Additional keyword arguments for subplots.

        Returns:
            tuple: A tuple containing the figure and axes.
        """
        if num_subplots == 1:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
            return fig, ax
        else:
            fig, axes = plt.subplots(nrows=kwargs.get('nrows', 1), 
                                     ncols=kwargs.get('ncols', num_subplots), 
                                     figsize=self.figsize)
            return fig, axes.flatten()

    def save_plot(self, fig, filename):
        """Saves the plot to a file.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            filename (str): The name of the file to save the plot to.
        """
        try:
            fig.savefig(filename, bbox_inches='tight')
            logger.info(f"Plot saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving plot to {filename}: {e}")
