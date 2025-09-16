import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

# Create a Matplotlib figure and axes
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y = np.cos(x)
ax.plot(x, y, color='red')
ax.set_title("Cosine Wave")
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")

# Display the Matplotlib figure in Streamlit
st.pyplot(fig)

records = [
    {"user_id": 1001, "name": "ZUBAIR HASSAN (B12,COLONY3 CHAKLALA GARRISON RAWALPINDI)", "cus_id": "83907", "email": "ali.ahmed@email.com", "phone": "0300-1234567", "city": "Lahore"},
    {"user_id": 1002, "name": "(FWO) PD HQ NBBIAP", "cus_id": "88594", "email": "fatima.khan@email.com", "phone": "0312-9876543", "city": "Karachi"},
    {"user_id": 1003, "name": "156 INDEPENDENT INFANTRY WORKSHOP COMPANY", "cus_id": "81155", "email": "usman.malik@email.com", "phone": "0333-4567890", "city": "Islamabad"},
    {"user_id": 1004, "name": "1st for Connect Pvt Ltd", "cus_id": "83025", "email": "ayesha.raza@email.com", "phone": "0345-1122334", "city": "Rawalpindi"},
    {"user_id": 1005, "name": "3G TECHNOLOGIES (GURANTEE) LIMITED", "cus_id": "82989", "email": "bilal.hassan@email.com", "phone": "0321-5566778", "city": "Faisalabad"},
    {"user_id": 1006, "name": "4 B INDUSTRIES", "cus_id": "99336", "email": "sanaullah@email.com", "phone": "0301-9988776", "city": "Sialkot"},
    {"user_id": 1007, "name": "4W Technologies (Pvt) Ltd", "cus_id": "81033", "email": "zainab.akhtar@email.com", "phone": "0335-4433221", "city": "Quetta"},
    {"user_id": 1008, "name": "A & Z OILS PVT. LTD.", "cus_id": "81626", "email": "imran.siddiqui@email.com", "phone": "0314-7778889", "city": "Multan"},
    {"user_id": 1009, "name": "Hina Sheikh", "cus_id": "82620", "email": "hina.sheikh@email.com", "phone": "0322-6543210", "city": "Sialkot"},
    {"user_id": 1010, "name": "A A BROTHER", "cus_id": "88737", "email": "omar.farooq@email.com", "phone": "0305-1357924", "city": "Sialkot"}
    ]

city_counts = {}
for r in records:
    city = r.get("city", "Unknown")
    city_counts[city] = city_counts.get(city, 0) + 1

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(city_counts.keys(), city_counts.values(), color="skyblue")
ax.set_title("Customer Count per Region")
ax.set_xlabel("Region")
ax.set_ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)