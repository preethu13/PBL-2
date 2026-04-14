"""
Notification Module for Flyover Enforcement System.

Sends violation notifications via:
1. Email (SMTP) with PDF fine report attached
2. SMS (Twilio) with violation summary

Includes a mock RTO database lookup for vehicle owner contact info.
"""

import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio not installed. SMS notifications disabled.")

# Import Violation
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.violation.logic_engine import Violation
except ImportError:
    try:
        from violation.logic_engine import Violation
    except ImportError:
        pass


# Mock RTO Database (in production, this queries Kerala State RTO API)
MOCK_RTO_DATABASE = {
    "KL07BJ4545": {"name": "John Thomas", "phone": "+919876543210", "email": "john@example.com"},
    "KL01AB1234": {"name": "Mary Joseph", "phone": "+919876543211", "email": "mary@example.com"},
    "KL10CD5678": {"name": "Rahul Menon", "phone": "+919876543212", "email": "rahul@example.com"},
    "KL05EF9012": {"name": "Anil Kumar", "phone": "+919876543213", "email": "anil@example.com"},
}


class Notifier:
    """
    Multi-channel notification sender for violation alerts.

    Supports email (with PDF attachment) and SMS notifications.
    Includes mock RTO database lookup for owner contact details.
    """

    def __init__(
        self,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        sender_email: str = "enforcement@kerala.gov.in",
        twilio_sid: str = "",
        twilio_token: str = "",
        twilio_from: str = "",
    ):
        """
        Initialize Notifier with email and SMS credentials.

        Args:
            smtp_host: SMTP server hostname.
            smtp_port: SMTP server port.
            smtp_user: SMTP authentication username.
            smtp_password: SMTP authentication password.
            sender_email: Sender email address.
            twilio_sid: Twilio Account SID.
            twilio_token: Twilio Auth Token.
            twilio_from: Twilio sender phone number.
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.sender_email = sender_email

        self.twilio_sid = twilio_sid
        self.twilio_token = twilio_token
        self.twilio_from = twilio_from
        self.twilio_client = None

        if TWILIO_AVAILABLE and twilio_sid and twilio_token:
            try:
                self.twilio_client = TwilioClient(twilio_sid, twilio_token)
                logger.info("Twilio client initialized")
            except Exception as e:
                logger.error(f"Twilio initialization failed: {e}")

        logger.info("Notifier initialized")

    def lookup_owner(self, plate: str) -> Dict:
        """
        Look up vehicle owner details from RTO database.

        NOTE: This is a MOCK implementation. In production, this
        would query the Kerala State RTO API (Parivahan/Sarathi).

        Args:
            plate: Number plate text.

        Returns:
            Dict with owner 'name', 'phone', 'email'.
        """
        owner = MOCK_RTO_DATABASE.get(plate)
        if owner:
            logger.info(f"RTO lookup for {plate}: {owner['name']}")
            return owner
        else:
            logger.info(f"RTO lookup for {plate}: not found (mock)")
            return {
                "name": "Vehicle Owner",
                "phone": "+910000000000",
                "email": "owner@example.com",
            }

    def send_email(
        self,
        to_addr: str,
        pdf_path: str,
        violation,
    ) -> bool:
        """
        Send violation notice email with PDF attachment.

        Args:
            to_addr: Recipient email address.
            pdf_path: Path to PDF fine report to attach.
            violation: Violation object for email body.

        Returns:
            True if email sent successfully, False otherwise.
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = to_addr
            msg['Subject'] = (
                f"Traffic Violation Notice - {violation.id} | "
                f"Kerala Motor Vehicles Department"
            )

            # Email body
            body = f"""
Dear Vehicle Owner,

This is an automated notification from the Kerala Motor Vehicles Department 
Automated Traffic Enforcement System.

A traffic violation has been recorded for your vehicle:

    Violation Reference: {violation.id}
    Number Plate: {violation.plate}
    Vehicle Type: {violation.vehicle_class}
    Violation: Entry of two-wheeler on restricted flyover
    Date & Time: {violation.timestamp}
    Location: {violation.location}
    Fine Amount: ₹{violation.fine_amount}
    Applicable Rule: {violation.rule}

Please find the detailed fine report attached as a PDF document.

Payment Instructions:
1. Visit your nearest Motor Vehicles Office
2. Pay online at https://parivahan.gov.in
3. Quote Reference Number: {violation.id}
4. Payment must be made within 30 days of this notice.

For queries, contact: Kerala MVD Helpline - 1800-425-1530

This is a system-generated notice. Please do not reply to this email.

Regards,
Kerala Motor Vehicles Department
Automated Traffic Enforcement System
"""
            msg.attach(MIMEText(body, 'plain'))

            # Attach PDF
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as f:
                    pdf_attachment = MIMEApplication(f.read(), _subtype='pdf')
                    pdf_attachment.add_header(
                        'Content-Disposition', 'attachment',
                        filename=os.path.basename(pdf_path)
                    )
                    msg.attach(pdf_attachment)

            # Send email
            if self.smtp_user and self.smtp_password:
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)
                logger.info(f"Email sent to {to_addr} for violation {violation.id}")
                return True
            else:
                logger.warning(
                    f"Email NOT sent (SMTP credentials not configured). "
                    f"Would send to: {to_addr}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to send email to {to_addr}: {e}")
            return False

    def send_sms(self, phone: str, message: str) -> bool:
        """
        Send SMS notification via Twilio.

        Args:
            phone: Recipient phone number (E.164 format).
            message: SMS message text.

        Returns:
            True if SMS sent successfully.
        """
        if not TWILIO_AVAILABLE or self.twilio_client is None:
            logger.warning(
                f"SMS NOT sent (Twilio not configured). "
                f"Would send to: {phone} — {message[:50]}..."
            )
            return False

        try:
            sms = self.twilio_client.messages.create(
                body=message,
                from_=self.twilio_from,
                to=phone,
            )
            logger.info(f"SMS sent to {phone} (SID: {sms.sid})")
            return True

        except Exception as e:
            logger.error(f"Failed to send SMS to {phone}: {e}")
            return False

    def notify_violation(self, violation, pdf_path: str = "") -> Dict:
        """
        Send all notifications for a violation.

        1. Look up vehicle owner
        2. Send email with PDF attachment
        3. Send SMS summary

        Args:
            violation: Violation object.
            pdf_path: Path to PDF report.

        Returns:
            Dict with 'email_sent' and 'sms_sent' boolean flags.
        """
        result = {"email_sent": False, "sms_sent": False}

        # Look up owner
        owner = self.lookup_owner(violation.plate)

        # Send email
        if owner.get("email"):
            result["email_sent"] = self.send_email(
                owner["email"],
                pdf_path or violation.pdf_path,
                violation,
            )

        # Send SMS
        if owner.get("phone"):
            sms_message = (
                f"Kerala MVD Notice: Traffic violation {violation.id} recorded for "
                f"vehicle {violation.plate} at {violation.location} on "
                f"{violation.timestamp}. Fine: Rs.{violation.fine_amount}. "
                f"Pay within 30 days at parivahan.gov.in. "
                f"Ref: {violation.id}"
            )
            result["sms_sent"] = self.send_sms(owner["phone"], sms_message)

        logger.info(
            f"Notifications for {violation.id}: "
            f"email={'✓' if result['email_sent'] else '✗'}, "
            f"sms={'✓' if result['sms_sent'] else '✗'}"
        )
        return result
