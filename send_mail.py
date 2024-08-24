import smtplib
import certifi
certifi.where()
import ssl
from email.message import EmailMessage

# Define email sender and receiver
email_sender = "garbagecollection21@gmail.com"
email_password = "rhagkxhdbqqhpdii"
email_receiver = "ritika2002.dps@gmail.com"

# Set the subject and body of the email
subject = 'Waste Detection using Drone'
body = 'Here Find the attachment of what the drone saw.'

em = EmailMessage()
em['From'] = email_sender
em['To'] = email_receiver
em['Subject'] = subject
attach = "output.csv"
em.set_content(body)

# Add SSL (layer of security)
context = ssl.create_default_context()

# Log in and send the email
with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
    smtp.login(email_sender, email_password)
    smtp.sendmail(email_sender, email_receiver, em.as_string())