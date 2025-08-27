from typing import Dict, Any
from utils.helpers import load_json_file

class RefundCalculator:
    def __init__(self, config_path: str = "data/config.json"):
        self.config = load_json_file(config_path)
        
    # Calculate refund based on policy and parameters
    def compute_refund(self, params: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            'refund_amount': 0,
            'applied_rules': [],
            'notes': []
        }
        category = params.get('category', 'default')
        # category specific return window
        return_windows = self.config['return_window_days_by_category']
        return_window = return_windows.get(category, return_windows['default'])

        if params['days_since_delivery'] > return_window:
            result['applied_rules'].append(f'Past {return_window}-day return window')
            result['notes'].append('No refund available - past return window')
            return result

        # category-specific restocking fee config
        category_config = self.config['restocking_fees'].get(
            category, self.config['restocking_fees']['default']
        )

        condition = 'opened' if params['opened'] else 'sealed'
        restocking_rate = category_config[condition]
        restocking_fee = params['purchase_price'] * restocking_rate

        refund_amount = params['purchase_price'] - restocking_fee

        result['refund_amount'] = round(refund_amount, 2)
        result['applied_rules'].append(f'{category.title()} item, {condition}')

        if restocking_fee > 0:
            result['applied_rules'].append(f'{restocking_rate*100:.0f}% restocking fee applied')
            result['notes'].append(f'Restocking fee: ${restocking_fee:.2f}')
        else:
            result['notes'].append('No restocking fee for sealed items')
        return result
    
