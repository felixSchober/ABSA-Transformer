from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

class Preferences(object):
    """
    Stores auto-saving preferences
    """
    path = './prefs.json'
    def __init__(self, **kwargs):
        if os.path.exists(Preferences.path):
            with open(Preferences.path, 'r')  as f:
                super(Preferences, self).__setattr__('prefs', json.load(f))
        else:
            super(Preferences, self).__setattr__('prefs', kwargs)
            
    def defaults(self, **kwargs):
        self.prefs.update(kwargs)

    def __repr__(self):
        return '\n'.join('{}={}'.format(k, v) for k,v in self.prefs.items())

    def __getattr__(self, name):
        return self.prefs[name]
    
    def __setattr__(self, name, value):
        
        self.prefs[name] = value
        
        # Serialize
        with open(Preferences.path, 'w') as f:
            json.dump(self.prefs, f)


# Add default preferences here
PREFERENCES = Preferences(
    overwrite_model_dir = False
)