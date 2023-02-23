purchases_cardholder:
ifdef n
	python3 purchases_cardholder.py $(n)
else
	python3 purchases_cardholder.py
endif

onsite_valid:
python3 onsite_transactions_validation.py

online_valid:
python3 online_transactions_validation.py

onsite_state:
python3 onsite_final_state.py

online_state:
python3 online_final_state.py
