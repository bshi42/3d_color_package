"""
Color theme management for InterDeCA module.

This module provides centralized color theme management with support for
both light and dark themes, automatically detecting Slicer's current theme.
"""

import qt


class ColorTheme:
    """Centralized color theme management for InterDeCA"""
    
    @staticmethod
    def getTheme():
        """Get color theme based on Slicer's current theme"""
        # Check if Slicer is in dark mode
        palette = qt.QApplication.palette()
        isDarkMode = palette.color(qt.QPalette.Window).lightness() < 128
        
        if isDarkMode:
            return ColorTheme.getDarkTheme()
        else:
            return ColorTheme.getLightTheme()
    
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
