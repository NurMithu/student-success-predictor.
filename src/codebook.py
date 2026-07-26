"""
Human-readable labels for the coded categorical columns in the UCI
"Predict Students' Dropout and Academic Success" dataset.

These mappings are used only by the Streamlit UI so users can pick
meaningful options (e.g. "Nursing") instead of raw integer codes (e.g. 9500).
Source: dataset documentation (Realinho et al., 2021) / UCI variable table.
"""

MARITAL_STATUS = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto union",
    6: "Legally separated",
}

APPLICATION_MODE = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2 (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)",
}

COURSE = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)",
}

DAYTIME_EVENING = {1: "Daytime", 0: "Evening"}

PREVIOUS_QUALIFICATION = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle or equivalent",
    38: "Basic education 2nd cycle",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)",
}

# Parents' qualification / occupation share the same broad coding scheme;
# for the UI we expose a simplified, readable subset (falls back to "Other / code N" if missing).
PARENT_QUALIFICATION = {
    1: "Secondary education - 12th year",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    9: "12th year - not completed",
    11: "7th year (old)",
    19: "Basic education 3rd cycle",
    27: "Incomplete secondary education",
    29: "9th year - not completed",
    30: "8th year of schooling",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read without 4th year",
    37: "Basic education 1st cycle",
    38: "Basic education 2nd cycle",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)",
    44: "Higher education - doctorate (3rd cycle)",
}

PARENT_OCCUPATION = {
    0: "Student",
    1: "Legislative / Executive / Director",
    2: "Intellectual / Scientific professional",
    3: "Technician / Associate professional",
    4: "Administrative staff",
    5: "Personal services / Security / Sales",
    6: "Farmer / Agriculture / Fishery",
    7: "Industry / Construction / Craftsman",
    8: "Machine operator",
    9: "Unskilled worker",
    10: "Armed forces",
    90: "Other situation",
    99: "(blank) / Not specified",
}

NATIONALITY = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldovan",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian",
}

YES_NO = {1: "Yes", 0: "No"}
GENDER = {1: "Male", 0: "Female"}


def label_for(mapping: dict, code, default_prefix="Code"):
    """Return a human-readable label for a code, or a generic fallback."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return str(code)
    return mapping.get(code, f"{default_prefix} {code}")


def options_for_select(mapping: dict):
    """Return (label, code) tuples sorted by label for use in a selectbox."""
    return sorted([(label, code) for code, label in mapping.items()], key=lambda x: x[0])
