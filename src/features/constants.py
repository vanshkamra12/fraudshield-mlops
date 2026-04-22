# Feature lists derived from EDA (notebooks/01_eda.ipynb)
# Edit this file to add/remove features — pipeline.py reads from here

# Drop these — >80% missing, no recoverable signal
COLS_TO_DROP = [
    'TransactionID',
    'dist2',
    'D6', 'D7', 'D9', 'D12', 'D13', 'D14',
    'id_03', 'id_04', 'id_07', 'id_08', 'id_18',
    'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27',
]

# Top 50 V-features by |correlation| with isFraud (from EDA section 7)
V_FEATURES = [
    'V45', 'V44', 'V86', 'V87', 'V52', 'V51', 'V40', 'V39', 'V38', 'V43',
    'V79', 'V42', 'V94', 'V74', 'V33', 'V17', 'V18', 'V81', 'V93', 'V92',
    'V82', 'V83', 'V75', 'V76', 'V77', 'V78', 'V80', 'V91', 'V12', 'V13',
    'V34', 'V35', 'V36', 'V37', 'V53', 'V54', 'V55', 'V56', 'V57', 'V70',
    'V95', 'V96', 'V97', 'V126', 'V127', 'V128', 'V130', 'V131', 'V307', 'V308',
]

# Encode these with target encoding (fitted on train only — no leakage)
TARGET_ENCODE_COLS = ['card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'ProductCD']

# Numeric features to median-impute (medians computed on train set only)
MEDIAN_IMPUTE_COLS = [
    'card1', 'card2', 'card3', 'card5',
    'addr1', 'addr2', 'dist1',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15',
    'id_01', 'id_02', 'id_05', 'id_06', 'id_11', 'id_13', 'id_17', 'id_19', 'id_20',
]

# M-flags — binary categorical (T/F/NaN)
M_COLS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

# Time-based train/test split boundary (day index from TransactionDT // 86400)
# Dataset spans day 1-182; day 145 gives ~80/20 split
TRAIN_TEST_SPLIT_DAY = 145
