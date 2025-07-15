import streamlit as st
import mysql.connector
import barcode
from barcode.writer import ImageWriter
import os

# Ensure barcode folder exists
if not os.path.exists("static/barcodes"):
    os.makedirs("static/barcodes")

# Generate 8-digit numeric ID manually (not random)
def generate_user_id():
    id = ""
    seed = 937
    for i in range(8):
        seed = (seed * 7 + i) % 10
        id = id + str(seed)
    return id

# Streamlit UI
st.title("Metro Card Registration")

# Registration form
with st.form("registration_form"):
    name = st.text_input("Enter your name")
    address = st.text_input("Enter your address")
    contact = st.text_input("Enter your contact number")
    submit = st.form_submit_button("Register and Generate E-Card")

# Only run below when form is submitted
if submit:
    user_id = generate_user_id()

    # Save to MySQL
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="0mm...123",
        database="metrodb"
    )
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (id, name, address, contact) VALUES (%s, %s, %s, %s)",
                   (user_id, name, address, contact))
    conn.commit()
    cursor.close()
    conn.close()

    # Generate barcode
    barcode_class = barcode.get_barcode_class('code128')
    barcode_file_path = "static/barcodes/" + user_id
    barcode_obj = barcode_class(user_id, writer=ImageWriter())
    barcode_obj.save(barcode_file_path)

    # Show Metro Card
    st.success("Registration Successful!")
    st.subheader("Your Metro Card")
    st.write("Name: " + name)
    st.write("Address: " + address)
    st.write("Contact: " + contact)
    st.write("Card ID: " + user_id)
    st.image(barcode_file_path + ".png", caption="Your Barcode")
