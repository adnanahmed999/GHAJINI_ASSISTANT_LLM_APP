"""
Creates Sanjay's document folder: one JSON file per note or document in data/docs/.

Sanjay forgets everything (like Ghajini), so he saves a note for everything:
doctor visits, bills, IDs, passwords, warranties, people, travel, money.

The documents the demo depends on are written by hand (PLANTED).
Everyday documents (EVERYDAY) are added so that search is actually needed.
Every document is unique.

Run once:  python make_docs.py
Then:      python rag.py --rebuild     (re-create the embeddings for the new documents)
"""

import json
import re
from datetime import date

from config import DOCS_DIR

# Each document: (date, type, title, text, due)
#   due = a date something must be done by (expiry, renewal, bill due), or None

# ------------------------------------------------------------------
# Demo documents (do not change these without running check_demo.py again)
# ------------------------------------------------------------------
PLANTED = [
    # "Am I allergic to anything?"  -> never uses the word "allergy" (Ctrl+F misses it, RAG finds it)
    ("2024-03-12", "Medical", "Doctor visit – Dr. Mehta, City Clinic",
     "Bad reaction after eating peanut chutney at the office canteen. My throat swelled up, I got red rashes "
     "all over my body and I could not breathe properly. Dr. Mehta said it was a severe reaction (anaphylaxis). "
     "Never eat peanuts or anything cooked in groundnut oil. Always carry the EpiPen injection in my bag.",
     None),

    # "What's my Wi-Fi password?"  -> the note says "router key"
    ("2025-06-21", "Home", "Home internet setup – Airtel Xstream",
     "Airtel technician installed the new router in the living room today. Network name: Sanjay_Home_5G. "
     "Router key: tiger@123. Plan: 300 Mbps, ₹999 per month. Customer care number: 121.",
     None),

    # "Is my passport still valid?"  -> needs today's date (tool)
    ("2016-11-15", "ID", "Passport",
     "Indian passport, number Z1234567 (sample). Issued at RPO Mumbai on 15 Nov 2016. Valid till 14 Nov 2026. "
     "Original kept in the brown file in the cupboard, photocopy in the laptop bag.",
     "2026-11-14"),

    # "When does my bike insurance end?"  -> note says "vehicle policy"
    ("2025-12-04", "Vehicle", "Vehicle policy renewed – Bajaj Allianz",
     "Renewed the comprehensive policy for my Pulsar 150 (MH-02-AB-1234). Premium paid: ₹2,350. "
     "Policy number OG-25-1234 (sample). Cover runs till 3 Dec 2026. Claim helpline: 1800-209-5858.",
     "2026-12-03"),

    # "Is my laptop still under warranty?"  -> date tool
    ("2024-11-02", "Warranty", "Laptop bill – Dell Inspiron from Croma",
     "Bought a Dell Inspiron 15 laptop from Croma, Andheri for ₹58,990. Invoice number CR-88213. "
     "2-year onsite warranty, so it is covered till 2 Nov 2026. Service centre: Dell, Marol.",
     "2026-11-02"),

    # "What should I gift Kalpana for her birthday?"  -> two notes disagree, the newer one wins
    ("2025-11-20", "People", "Kalpana – things to remember",
     "Kalpana's birthday is on 22 January. She loves chocolate truffle cake from Theobroma. "
     "Favourite flowers: white lilies. She hates horror movies, never take her to one.",
     None),
    ("2026-08-18", "People", "Kalpana – health update",
     "Kalpana's doctor told her to stop sugar completely. No cakes, no mithai, no chocolates for her now. "
     "Better gift ideas: a good book, an indoor plant, or a handwritten letter.",
     None),

    # "Where are my bank locker keys?"
    ("2023-05-10", "Money", "Bank locker – SBI Andheri West",
     "Took a locker at SBI Andheri West branch. Locker number 214. Yearly rent ₹2,000, auto-debited in April. "
     "The locker key is in the small blue box under my bed. Branch manager: Mr. Kulkarni.",
     None),

    # "What's my blood group?"
    ("2025-02-08", "Medical", "Blood test report – Apollo Diagnostics",
     "Full body check-up at Apollo Diagnostics. Blood group: B positive. Haemoglobin 14.2, vitamin D slightly low. "
     "Doctor asked me to take vitamin D sachets once a week for 8 weeks.",
     None),

    # "Who is my landlord?"  -> note says "flat owner"
    ("2026-10-05", "Home", "Flat payment – October 2026",
     "Paid ₹15,000 for the flat to Mr. Sharma (flat owner) through bank transfer. "
     "His number: 98200 12345. Maintenance is included. Deposit of ₹50,000 is with him.",
     None),

    # "What do I need to renew soon?"  -> upcoming_deadlines tool (MCP) also finds these
    ("2026-10-12", "Bills", "Electricity bill – Adani Electricity, October 2026",
     "Electricity bill for October 2026: ₹1,840 for 262 units. Due date 25 Oct 2026. Not paid yet. "
     "Pay through the Adani app or GPay.",
     "2026-10-25"),
    ("2026-04-30", "Vehicle", "Bike pollution certificate (PUC)",
     "Got the pollution under control certificate for the Pulsar at the petrol pump near Andheri station. "
     "Paid ₹100. Valid for 6 months, till 30 Oct 2026. Keep the paper in the bike's side box.",
     "2026-10-30"),
]

# ------------------------------------------------------------------
# Everyday documents: nothing to do with the demo questions.
# Keep them away from the demo topics (peanuts/reactions, router, passport, bike policy,
# laptop warranty, Kalpana's gifts, locker, blood group, flat payment).
# ------------------------------------------------------------------
EVERYDAY = [
    # medical (plain, nothing that looks like a reaction)
    ("2023-08-14", "Medical", "Dentist – Dr. Shah", "Root canal on the lower left tooth. Two sittings, total ₹6,500. Next check-up after 6 months. Use the soft toothbrush.", None),
    ("2024-07-02", "Medical", "Fever – Dr. Mehta", "Viral fever for three days, 101°F. Took paracetamol 650 twice a day. Rest and drink lots of water. Better by Friday.", None),
    ("2025-09-19", "Medical", "Eye test – Lenskart", "Eye test: right eye -1.25, left eye -1.00. Ordered new specs with blue-cut lenses, ₹3,200. Ready in 5 days.", None),
    ("2026-01-15", "Medical", "Back pain – physiotherapy", "Lower back pain from sitting too long. Physio at Joy Clinic, 6 sessions of ₹500. Do the cat-camel stretch every morning.", None),
    ("2025-05-22", "Medical", "Health insurance – Star Health", "Family floater mediclaim of ₹5 lakh with Star Health. Premium ₹14,200 a year. Renewal on 1 March every year. Cashless at Nanavati and Kokilaben.", None),
    ("2026-06-10", "Medical", "Vaccination record", "Took the tetanus booster at the municipal dispensary after a small cut from a rusty gate. Next booster after 10 years.", None),
    ("2024-12-03", "Medical", "Sleep notes", "Not sleeping well. Doctor said no phone after 11 pm, no tea after 6 pm, and walk 30 minutes daily.", None),

    # IDs and documents
    ("2019-04-20", "ID", "PAN card", "PAN card number ABCDE1234F (sample). Original in the brown file, photo on Google Drive in the 'IDs' folder.", None),
    ("2018-09-11", "ID", "Aadhaar card", "Aadhaar number XXXX-XXXX-4321 (sample). Linked to my mobile number. Address updated to the Andheri flat in 2023.", None),
    ("2021-02-17", "ID", "Driving licence", "Driving licence for two-wheeler and car, issued by RTO Andheri. Valid till 16 Feb 2041. Digital copy in DigiLocker.", None),
    ("2023-03-03", "ID", "Voter ID", "Voter ID card collected from the BMC office. Polling booth: St. Xavier's School, room 4.", None),
    ("2024-06-28", "ID", "DigiLocker login", "All important IDs are saved in DigiLocker. Login is through my mobile number and OTP. No separate app PIN.", None),

    # money
    ("2023-01-09", "Money", "Fixed deposit – HDFC", "Opened an FD of ₹1,00,000 at 7.1% for 3 years. Matures on 9 Jan 2026. Interest goes to the savings account.", None),
    ("2026-01-10", "Money", "FD matured – HDFC", "The HDFC FD matured today. ₹1,23,600 credited to savings. Moved ₹1,00,000 into a mutual fund SIP plan.", None),
    ("2026-01-12", "Money", "SIP – index fund", "Started a monthly SIP of ₹5,000 in a Nifty 50 index fund on the 5th of every month. Goal: emergency fund of ₹3 lakh.", None),
    ("2025-07-31", "Money", "Income tax return filed", "Filed the income tax return for last year on the e-filing portal. Refund of ₹4,200 expected. Acknowledgement saved in the 'Tax' folder.", None),
    ("2024-10-18", "Money", "Lent money to Rahul", "Lent ₹8,000 to Rahul for his college fees. He said he will return it by Diwali. (Returned on 30 Oct.)", None),
    ("2025-03-15", "Money", "Credit card – Axis", "Axis credit card bill is generated on the 15th, pay by the 5th. Auto-pay set for the full amount.", None),
    ("2026-02-27", "Money", "Monthly budget", "Budget for March: flat ₹15,000, food ₹6,000, travel ₹2,500, phone and internet ₹1,300, savings ₹8,000, fun ₹3,000.", None),
    ("2025-11-02", "Money", "UPI ID", "My UPI ID is sanjay@okhdfc (sample). The UPI PIN is NOT written anywhere, I know it.", None),

    # home
    ("2023-02-01", "Home", "Moved to the Andheri flat", "Moved into Flat 302, Shanti Apartments, Andheri West. 1 BHK. Society secretary: Mrs. Pinto, flat 101.", None),
    ("2024-04-14", "Home", "Fridge – LG double door", "Bought an LG 260 L fridge from Vijay Sales for ₹27,500. Compressor warranty 10 years, other parts 1 year.", None),
    ("2025-01-25", "Home", "Washing machine repair", "Washing machine was not draining. Repairman Imran fixed the pump for ₹1,200. His number: 98190 55555.", None),
    ("2024-09-08", "Home", "Maid – Sunita tai", "Sunita tai comes at 8 am for cleaning and dishes. ₹3,500 a month, paid on the 1st. Off on Sundays.", None),
    ("2025-08-02", "Home", "Gas cylinder booking", "Book the HP gas cylinder through the app or by calling 1906. One cylinder lasts about 6 weeks.", None),
    ("2026-03-20", "Home", "Water purifier service", "Aquaguard service done: filter and UV lamp changed, ₹2,800. Next service after 12 months.", None),
    ("2025-10-11", "Home", "Spare house key", "Spare key of the flat is with Mrs. Pinto in flat 101. Another one is in my office drawer.", None),
    ("2026-07-07", "Home", "Plumber", "Bathroom tap was leaking. Plumber Raju fixed it for ₹400. Good guy, comes on time. Number: 99300 11111.", None),

    # vehicle
    ("2022-06-18", "Vehicle", "Bought the Pulsar", "Bought a Bajaj Pulsar 150, black, from Shree Bajaj, Andheri. Number MH-02-AB-1234. ₹1,12,000 on-road.", None),
    ("2025-04-06", "Vehicle", "Bike service", "Bike serviced at Shree Bajaj: oil change, brake pads, chain cleaning. ₹1,650. Next service at 18,000 km.", None),
    ("2026-02-14", "Vehicle", "Traffic fine", "Got a ₹500 fine for parking near the station. Paid online on the Mahatraffic app. Park in the paid lot next time.", None),
    ("2025-12-20", "Vehicle", "Tyre change", "Changed the rear tyre of the Pulsar to MRF, ₹2,100. Tyre pressure: 25 front, 32 rear.", None),

    # work
    ("2024-01-08", "Work", "Joined TechNova", "Joined TechNova Solutions as a junior developer. Office: Powai, 5th floor. Manager: Priya Nair. Employee ID TN-4521.", None),
    ("2025-04-01", "Work", "Appraisal", "Got a 12% hike and moved to developer grade 2. New project: billing app for a retail client.", None),
    ("2026-05-18", "Work", "Office leave policy", "24 paid leaves a year, 8 sick leaves. Apply on the HR portal at least 3 days before.", None),
    ("2025-10-28", "Work", "Office Wi-Fi is not mine", "Note to self: the office network belongs to TechNova IT, ask the IT desk on the 2nd floor if it stops working.", None),
    ("2026-09-02", "Work", "Team lunch", "Team lunch at Mainland China, Powai for Priya's farewell. Everyone paid ₹900.", None),
    ("2024-08-19", "Work", "Laptop rules at office", "Office laptop must be locked with Win+L when I leave the desk. Never install personal apps on it.", None),
    ("2026-08-25", "Work", "Project deadline", "Billing app release went live on 25 Aug. Client happy. Next phase starts in November.", None),

    # people
    ("2023-06-02", "People", "Rahul – college friend", "Rahul Verma, college friend from Pune. Works at an ad agency in Bandra. Supports CSK, never talk cricket with him.", None),
    ("2024-02-10", "People", "Mom and Dad", "Mom's number 98201 00001, Dad's 98201 00002. Call them every Sunday evening. Dad's knee check-up every 3 months.", None),
    ("2025-03-30", "People", "Neighbour – Mr. Iyer", "Mr. Iyer in 304, retired bank officer. Helpful with forms and bank work. Likes filter coffee.", None),
    ("2026-04-11", "People", "Priya Nair – manager", "Priya likes short updates in the morning stand-up. Send the weekly report every Friday by 5 pm.", None),
    ("2024-05-05", "People", "Cousin Amit's wedding", "Amit's wedding in Nashik on 5 May. Gave ₹5,001 in the envelope. Next family function: Diwali at Pune.", None),
    ("2025-07-12", "People", "Doctor contacts", "Dr. Mehta (family doctor) 98210 22222, Dr. Shah (dentist) 98210 33333, Joy Clinic physio 98210 44444.", None),

    # travel
    ("2024-12-24", "Travel", "Goa trip", "Goa trip with Rahul and friends, 24–28 Dec. Stayed at a guest house in Anjuna, ₹1,800 a night. Rented a scooter.", None),
    ("2025-05-16", "Travel", "Train to Pune", "Deccan Queen to Pune, coach C3, seat 45. PNR 412-5566778 (sample). ₹410.", None),
    ("2026-03-28", "Travel", "Manali trip plan", "Manali trip in May: flight to Chandigarh, then Volvo bus. Pack warm jacket, gloves, power bank, and ID cards.", None),
    ("2026-05-12", "Travel", "Manali hotel", "Hotel Snow View, Old Manali, 12–16 May. Booking ID MMT-90876. Breakfast included.", None),
    ("2025-12-31", "Travel", "Lonavala day trip", "New year's eve day trip to Lonavala with the office gang. Tiger Point at sunset was the best part.", None),

    # shopping and warranties (not the laptop)
    ("2025-02-20", "Warranty", "Phone – Samsung Galaxy", "Bought a Samsung Galaxy A55 for ₹38,000 on Amazon. 1-year warranty. Screen guard and back cover from the shop next door.", None),
    ("2024-11-25", "Warranty", "Mixer grinder – Bajaj", "Bajaj mixer grinder from DMart, ₹2,900. 2-year warranty on the motor. Bill is stuck inside the kitchen cupboard door.", None),
    ("2025-06-08", "Warranty", "Headphones – boAt", "boAt Rockerz headphones, ₹1,499 on Flipkart. 1-year warranty, claim through the boAt website.", None),
    ("2026-01-30", "Warranty", "Smartwatch", "Noise smartwatch gift from Mom. Charger is the magnetic one in the drawer next to the TV.", None),
    ("2025-09-05", "Shopping", "Office shoes", "Bought black formal shoes from Bata, size 9. ₹2,499. Size 9 fits, 8 is tight.", None),
    ("2026-06-22", "Shopping", "Clothes sizes", "Shirt size: 40 (M). Jeans: 32 waist, 32 length. T-shirt: M. Shoes: 9.", None),

    # subscriptions
    ("2025-01-04", "Bills", "Mobile plan – Jio", "Jio prepaid plan ₹299 a month, 2 GB a day. Recharge on the 4th. Number 70450 12345 (sample).", None),
    ("2025-03-01", "Bills", "Netflix subscription", "Netflix basic plan ₹199 a month on the credit card. Profile names: Sanjay, Guest.", None),
    ("2025-08-15", "Bills", "Gym membership – Gold's Gym", "Gold's Gym, Andheri. 6 months for ₹9,000. Timing: 7–8 am. Locker number 56.", None),
    ("2026-02-02", "Bills", "Newspaper", "Times of India delivered daily by Shankar. ₹450 a month, paid in cash at the end of the month.", None),

    # learning and hobbies
    ("2024-03-25", "Learning", "Python course", "Started the Python course on Coursera. Finished week 3: loops and functions. Practice 30 minutes daily.", None),
    ("2025-11-14", "Learning", "AWS certification", "Booked the AWS Cloud Practitioner exam. Study plan: 2 hours on weekends. Exam fee ₹8,500.", None),
    ("2026-01-20", "Learning", "AWS exam result", "Passed the AWS Cloud Practitioner exam with 812 marks. Certificate saved in the 'Certificates' folder.", None),
    ("2025-04-19", "Hobby", "Guitar practice", "Learning guitar from YouTube. Chords done: G, C, D, Em. Song in progress: Tum Hi Ho.", None),
    ("2024-10-02", "Hobby", "Books to read", "Books to read: Atomic Habits, The Alchemist, Ikigai, Rich Dad Poor Dad. Done: The Alchemist.", None),
    ("2025-06-30", "Hobby", "Movies watched", "Movies watched this year: 3 Idiots (again), Dangal, Taare Zameen Par, Lagaan. Best: Lagaan.", None),
    ("2026-09-14", "Hobby", "Running", "Ran 5 km at Juhu beach in 32 minutes. Target: 10 km run in January.", None),

    # recipes and food
    ("2025-02-03", "Recipe", "Mom's dal tadka", "Pressure cook 1 cup toor dal with turmeric and salt, 3 whistles. Tadka: ghee, jeera, garlic, red chilli, hing. Add coriander.", None),
    ("2025-10-07", "Recipe", "Quick poha", "Wash 2 cups poha. Fry mustard seeds, curry leaves, onion, green chilli, potato. Add poha, turmeric, salt, lemon. Top with sev.", None),
    ("2026-04-02", "Recipe", "Masala chai", "1 cup water, 1 cup milk, 2 tsp tea, crushed ginger, 2 cardamom. Boil for 4 minutes. Strain.", None),
    ("2024-08-29", "Food", "Tiffin service", "Annapurna tiffin service, ₹3,000 a month for lunch. Call by 10 am to cancel for the day. Number 98192 77777.", None),

    # notes to self and misc
    ("2023-07-21", "Note", "Where I keep things", "Brown file: IDs and certificates. Blue box under bed: small keys. Office drawer: spare charger and spare flat key.", None),
    ("2024-01-01", "Note", "New year goals", "Goals: save ₹2 lakh, learn Python, run a half marathon, call parents every Sunday, sleep before midnight.", None),
    ("2025-01-01", "Note", "New year goals – again", "Goals: get AWS certified, go to Manali, read 6 books, keep the bike clean, stop losing keys.", None),
    ("2024-06-15", "Note", "Phone backup", "Phone photos back up to Google Photos automatically. WhatsApp backup every night at 2 am.", None),
    ("2025-12-08", "Note", "Emergency contacts", "In an emergency call Rahul 98192 88888 or Mom 98201 00001. Nearest hospital: Kokilaben, Andheri West.", None),
    ("2026-07-19", "Note", "Parking at office", "Office parking is in basement B2, slot 118. Parking sticker renewal every April.", None),
    ("2023-11-12", "Note", "Diwali shopping list", "Diya, rangoli colours, new kurta, gifts for Sunita tai and the watchman. Budget ₹6,000.", None),
    ("2025-08-30", "Note", "Ganpati at society", "Society Ganpati for 5 days. Contribution ₹1,001 given to Mrs. Pinto. Aarti at 8 pm.", None),
    ("2026-05-30", "Note", "Monsoon checklist", "Before monsoon: buy a raincoat, check the window seals, keep a spare umbrella in the office, cover the bike.", None),
]

# Monthly electricity bills (Nov 2025 – Sep 2026): already paid, each with its own amount
PAID_BILLS = [
    ("2025-11", 1520), ("2025-12", 1310), ("2026-01", 1240), ("2026-02", 1290), ("2026-03", 1610),
    ("2026-04", 2140), ("2026-05", 2480), ("2026-06", 1980), ("2026-07", 1720), ("2026-08", 1690),
    ("2026-09", 1760),
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50]


def all_documents() -> list[tuple]:
    docs = list(PLANTED) + list(EVERYDAY)
    for month, amount in PAID_BILLS:
        d = date.fromisoformat(month + "-12")
        units = amount // 7
        docs.append((
            d.isoformat(), "Bills", f"Electricity bill – Adani Electricity, {d.strftime('%B %Y')}",
            f"Electricity bill for {d.strftime('%B %Y')}: ₹{amount:,} for {units} units. Paid on time through GPay.",
            None,
        ))
    return docs


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for old in DOCS_DIR.glob("*.json"):
        old.unlink()

    docs = all_documents()
    for d, kind, title, text, due in docs:
        doc_id = f"{d}_{slug(title)}"
        record = {"id": doc_id, "date": d, "type": kind, "title": title, "text": text}
        if due:
            record["due"] = due
        (DOCS_DIR / f"{doc_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Created {len(docs)} documents ({len(PLANTED)} demo + {len(docs) - len(PLANTED)} everyday) in {DOCS_DIR}")


if __name__ == "__main__":
    main()
