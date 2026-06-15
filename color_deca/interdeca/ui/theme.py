"""Theme helpers for the InterDeCA Slicer UI."""

try:
    import qt
except ModuleNotFoundError:  # Allows standard-Python import of this module.
    qt = None


def _require_qt():
    if qt is None:
        raise RuntimeError("ColorTheme requires Slicer's qt module at runtime")
    return qt


class ColorTheme:
    """Centralized color theme management for InterDeCA.

    Provides consistent styling across the module interface with
    support for both light and dark themes that adapt to Slicer's
    current appearance settings.
    """

    @staticmethod
    def getTheme():
        """Gets color theme based on Slicer's current theme.

        Returns:
            dict: Theme colors appropriate for current mode
        """
        qt_module = _require_qt()
        # Checks if Slicer is in dark mode by examining window palette
        palette = qt_module.QApplication.palette()  # Gets application palette
        isDarkMode = palette.color(qt_module.QPalette.Window).lightness() < 128  # Determines if dark

        if isDarkMode:
            return ColorTheme.getDarkTheme()  # Returns dark theme colors
        else:
            return ColorTheme.getLightTheme()  # Returns light theme colors
    
    @staticmethod
    def getLightTheme():
        return {
            'primary': '#4CAF50',
            'primary_hover': '#45A049',
            'primary_pressed': '#3D8B40',
            'secondary': '#87CEEB',
            'secondary_hover': '#6BB6E8',
            'secondary_pressed': '#4FA8D8',
            'accent': '#9C27B0',
            'accent_hover': '#8E24AA',
            'accent_pressed': '#7B1FA2',
            'danger': '#FF6B6B',
            'danger_hover': '#FF5252',
            'danger_pressed': '#E53935',
            'neutral': '#607D8B',
            'neutral_hover': '#546E7A',
            'neutral_pressed': '#455A64',
            'text_primary': '#2C3E50',
            'text_secondary': 'palette(disabled-text)',
            'text_on_primary': 'white',
            'disabled_bg': '#CCCCCC',
            'disabled_text': '#666666'
        }
    
    @staticmethod
    def getDarkTheme():
        return {
            'primary': '#66BB6A',
            'primary_hover': '#5CB85C',
            'primary_pressed': '#4CAF50',
            'secondary': '#64B5F6',
            'secondary_hover': '#42A5F5',
            'secondary_pressed': '#2196F3',
            'accent': '#BA68C8',
            'accent_hover': '#AB47BC',
            'accent_pressed': '#9C27B0',
            'danger': '#EF5350',
            'danger_hover': '#E53935',
            'danger_pressed': '#D32F2F',
            'neutral': '#78909C',
            'neutral_hover': '#607D8B',
            'neutral_pressed': '#546E7A',
            'text_primary': '#FFFFFF',
            'text_secondary': 'palette(disabled-text)',
            'text_on_primary': 'white',
            'disabled_bg': '#424242',
            'disabled_text': '#9E9E9E'
        }
    
    @staticmethod
    def getButtonStyle(color_type='primary', disabled_style=True):
        """Generate button stylesheet with theme colors"""
        theme = ColorTheme.getTheme()
        
        style = f"""
        QPushButton {{
            background-color: {theme[color_type]};
            color: {theme['text_on_primary']};
            font-weight: bold;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            min-height: 20px;
        }}
        QPushButton:hover {{
            background-color: {theme[color_type + '_hover']};
        }}
        QPushButton:pressed {{
            background-color: {theme[color_type + '_pressed']};
        }}"""
        
        if disabled_style:
            style += f"""
        QPushButton:disabled {{
            background-color: {theme['disabled_bg']};
            color: {theme['disabled_text']};
        }}"""
        
        return style
    
    @staticmethod
    def getLabelStyle(style_type='secondary'):
        """Generate label stylesheet with theme colors"""
        theme = ColorTheme.getTheme()
        return f"QLabel {{ color: {theme['text_secondary']}; font-style: italic; }}"
    
    @staticmethod
    def getProgressLabelStyle():
        """Get progress label style"""
        return """
        QLabel { 
            color: palette(link); 
            font-weight: bold; 
        }"""
    
    @staticmethod
    def getHeaderStyle():
        """Get header/collapsible button style with theme-aware colors"""
        theme = ColorTheme.getTheme()
        return f"""
        ctkCollapsibleButton {{ 
            border: 1px solid palette(mid);
            border-radius: 4px;
            background-color: palette(window);
        }}
        /* Style the QToolButton which is the actual header/title */
        ctkCollapsibleButton QToolButton {{
            font-weight: bold; 
            color: {theme['text_primary']}; 
            background-color: palette(alternate-base);
            padding: 4px;
            border: none;
            text-align: left;
        }}
        /* Hover effect only on the header button */
        ctkCollapsibleButton QToolButton:hover {{
            background-color: palette(highlight);
            color: palette(highlighted-text);
        }}
        """
    
    @staticmethod
    def getSectionLabelStyle():
        """Get section label style with theme-aware colors"""
        theme = ColorTheme.getTheme()
        return f"""
        QLabel {{ 
            font-weight: bold; 
            color: {theme['text_primary']}; 
            margin-top: 10px;
        }}
        """
    
    @staticmethod
    def getStatusLabelStyle(status_type='neutral'):
        """Get status label style with theme-aware colors"""
        theme = ColorTheme.getTheme()
        if status_type == 'success':
            color = theme['primary']
        elif status_type == 'error':
            color = theme['danger']
        elif status_type == 'warning':
            color = '#ff8c00'  # Orange that works in both themes
        elif status_type == 'progress':
            color = theme['secondary']
        else:  # neutral
            color = theme['text_secondary']
        
        return f"""
        QLabel {{ 
            color: {color}; 
            font-style: italic;
        }}
        """
    
    @staticmethod
    def getSeparatorStyle():
        """Get separator line style with theme-aware colors"""
        return """
        QFrame { 
            color: palette(mid); 
            margin: 10px 0px; 
        }
        """
    
    @staticmethod
    def getValidationLabelStyle(validation_type='neutral'):
        """Get validation label style with theme-aware colors"""
        if validation_type == 'success':
            color = 'palette(positive)'
        elif validation_type == 'error':
            color = 'palette(negative)'
        elif validation_type == 'warning':
            color = '#ff8c00'  # Orange that works in both themes
        else:  # neutral
            color = 'palette(disabled-text)'
        
        return f"""
        QLabel {{ 
            color: {color}; 
            font-style: italic;
        }}
        """
    
    @staticmethod
    def getComboBoxStyle():
        """Get combobox/dropdown style with theme-aware colors (for both QComboBox and qMRMLNodeComboBox)"""
        return """
        QComboBox, qMRMLNodeComboBox {
            color: palette(window-text);
            background-color: palette(base);
            border: 1px solid palette(mid);
            border-radius: 3px;
            padding: 3px 5px;
        }
        QComboBox:on, qMRMLNodeComboBox:on {
            color: palette(window-text);
        }
        QComboBox:hover, qMRMLNodeComboBox:hover {
            border: 1px solid palette(highlight);
        }
        QComboBox:disabled, qMRMLNodeComboBox:disabled {
            color: palette(disabled-text);
            background-color: palette(window);
        }
        QComboBox::drop-down, qMRMLNodeComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView, qMRMLNodeComboBox QAbstractItemView {
            color: palette(window-text);
            background-color: palette(base);
            selection-background-color: palette(highlight);
            selection-color: palette(highlighted-text);
        }
        """
