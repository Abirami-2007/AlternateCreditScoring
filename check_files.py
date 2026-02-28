import os

files = {
    "Home Credit Main":     "data/home_credit/application_train.csv",
    "Bureau":               "data/home_credit/bureau.csv",
    "Bureau Balance":       "data/home_credit/bureau_balance.csv",
    "Installments":         "data/home_credit/installments_payments.csv",
    "POS Cash":             "data/home_credit/POS_CASH_balance.csv",
    "Credit Card":          "data/home_credit/credit_card_balance.csv",
    "Previous App":         "data/home_credit/previous_application.csv",
    "Give Me Some Credit":  "data/give_me_some_credit/cs-training.csv",
    "Lending Club":         "data/lending_club/loan.csv",
    "PKDD Loan":            "data/pkdd_czech/loan.csv",
    "PKDD Trans":           "data/pkdd_czech/trans.csv",
    "PKDD Account":         "data/pkdd_czech/account.csv",
}

all_ok = True

for name, path in files.items():
    exists = os.path.exists(path)
    if exists:
        size = f"{os.path.getsize(path)/1e6:.0f} MB"
        print(f"    {name:<25} {size}")
    else:
        print(f"    {name:<25} MISSING")
        all_ok = False

print()
print("ALL GOOD — ready to run!" if all_ok else "Fix missing files first")