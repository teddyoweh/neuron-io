RE_PATTERNS = {
    'email_address': r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$',
    'url': r'^(http|https)://[^\s/$.?#].[^\s]*$',
    'date_mmddyyyy': r'^(0[1-9]|1[0-2])/(0[1-9]|1\d|2\d|3[01])/((19|20)\d{2})$',
    'ipv4_address': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
    'phone_number_us': r'^\+?1?\d{10}$',
    'zip_code_us': r'^\d{5}(?:[-\s]\d{4})?$',
    'ssn_us': r'^\d{3}-\d{2}-\d{4}$',
    'credit_card_visa': r'^4[0-9]{12}(?:[0-9]{3})?$',
    'html_tag': r'<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)$',
    'hex_color_code': r'^#?([a-fA-F0-9]{6}|[a-fA-F0-9]{3})$',
    'time_hhmm_ampm': r'^(0[1-9]|1[0-2]):[0-5][0-9] (AM|PM)$',
    'file_extension': r'^[\w\-]+\.(\w+)$',
    'floating_point_number': r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$',
    'alphanumeric': r'^[a-zA-Z0-9]+$',
    'username': r'^[a-zA-Z0-9_]+$',
    'html_comment': r'<!--(.*?)-->',
    'positive_integer': r'^[1-9]\d*$',
    'sentence': r'^[A-Z][^.!?]*[.!?]$',
    'markdown_heading': r'^#{1,6}\s.*$',
    'alphabetic_word': r'^[a-zA-Z]+$',
    'non_alphanumeric': r'[^a-zA-Z0-9\s]'
}

