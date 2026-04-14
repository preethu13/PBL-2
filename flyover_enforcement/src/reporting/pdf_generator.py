"""
PDF Fine Report Generator for Flyover Enforcement System.

Generates official-looking fine notices using ReportLab with:
- Department header and logo placeholder
- Violation details table
- Annotated snapshot image
- QR code with violation ID
- Legal footer

Output: Professional PDF fine report ready for dispatch.
"""

import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch, cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, black, white, red
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable,
    )
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not installed. PDF generation disabled.")

try:
    import qrcode
    from io import BytesIO
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    logger.warning("qrcode not installed. QR codes disabled in PDF.")

# Import Violation class
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.violation.logic_engine import Violation
except ImportError:
    try:
        from violation.logic_engine import Violation
    except ImportError:
        from dataclasses import dataclass, field
        @dataclass
        class Violation:
            id: str = ""
            plate: str = ""
            vehicle_class: str = ""
            timestamp: str = ""
            snapshot_path: str = ""
            confidence: float = 0.0
            fine_amount: int = 500
            location: str = ""
            rule: str = ""
            status: str = "pending"
            pdf_path: str = ""
            bbox: tuple = field(default_factory=tuple)


class FineReportGenerator:
    """
    Generates PDF fine reports for traffic violations.

    Creates professional, official-looking notices similar to
    those issued by the Kerala Motor Vehicles Department.
    """

    def __init__(self, output_dir: str = "data/violations/reports"):
        """
        Initialize PDF Generator.

        Args:
            output_dir: Directory to save generated PDFs.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()
            logger.info(f"FineReportGenerator initialized (output={output_dir})")
        else:
            logger.error("ReportLab not available — PDF generation disabled")

    def _setup_custom_styles(self):
        """Define custom paragraph styles for the report."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=HexColor('#1a237e'),
            alignment=TA_CENTER,
            spaceAfter=12,
        ))
        self.styles.add(ParagraphStyle(
            name='SubTitle',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=HexColor('#333333'),
            alignment=TA_CENTER,
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name='ViolationHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#c62828'),
            alignment=TA_CENTER,
            spaceBefore=12,
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=HexColor('#666666'),
            alignment=TA_CENTER,
        ))

    def _generate_qr(self, data: str, size: int = 120) -> Optional[object]:
        """
        Generate a QR code image for the violation ID.

        Args:
            data: Data to encode in QR code.
            size: QR code image size in pixels.

        Returns:
            ReportLab Image object, or None if qrcode not available.
        """
        if not QR_AVAILABLE or not REPORTLAB_AVAILABLE:
            return None

        try:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=10,
                border=2,
            )
            qr.add_data(data)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")

            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)

            return RLImage(buffer, width=size, height=size)
        except Exception as e:
            logger.error(f"QR generation failed: {e}")
            return None

    def generate_pdf(self, violation: Violation) -> str:
        """
        Generate a complete PDF fine report.

        PDF Contents:
        1. Header with department name and logo placeholder
        2. "TRAFFIC VIOLATION NOTICE" title
        3. Violation details table
        4. Snapshot image of the violation
        5. QR code with violation ID
        6. Legal footer

        Args:
            violation: Violation object with all details.

        Returns:
            Path to generated PDF file.
        """
        if not REPORTLAB_AVAILABLE:
            logger.error("Cannot generate PDF — ReportLab not installed")
            return ""

        pdf_filename = f"{violation.id}_fine_report.pdf"
        pdf_path = os.path.join(self.output_dir, pdf_filename)

        try:
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=A4,
                rightMargin=1.5 * cm,
                leftMargin=1.5 * cm,
                topMargin=1.5 * cm,
                bottomMargin=1.5 * cm,
            )

            elements = []

            # === HEADER ===
            elements.append(Paragraph(
                "GOVERNMENT OF KERALA",
                self.styles['ReportTitle']
            ))
            elements.append(Paragraph(
                "Kerala Motor Vehicles Department",
                self.styles['SubTitle']
            ))
            elements.append(Paragraph(
                "Automated Traffic Enforcement System",
                self.styles['SubTitle']
            ))

            # Horizontal rule
            elements.append(Spacer(1, 6))
            elements.append(HRFlowable(
                width="100%", thickness=2, color=HexColor('#1a237e')
            ))
            elements.append(Spacer(1, 12))

            # === VIOLATION NOTICE TITLE ===
            elements.append(Paragraph(
                "⚠ TRAFFIC VIOLATION NOTICE ⚠",
                self.styles['ViolationHeader']
            ))
            elements.append(Spacer(1, 12))

            # === NOTICE REFERENCE ===
            ref_data = [
                ['Notice Reference:', violation.id],
                ['Date of Issue:', datetime.now().strftime("%d-%m-%Y")],
            ]
            ref_table = Table(ref_data, colWidths=[150, 300])
            ref_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TEXTCOLOR', (0, 0), (-1, -1), black),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(ref_table)
            elements.append(Spacer(1, 16))

            # === VIOLATION DETAILS TABLE ===
            elements.append(Paragraph(
                "<b>Violation Details</b>",
                self.styles['Heading3']
            ))
            elements.append(Spacer(1, 6))

            details_data = [
                ['Field', 'Details'],
                ['Number Plate', violation.plate],
                ['Vehicle Type', violation.vehicle_class.title()],
                ['Violation', 'Entry of two-wheeler on restricted flyover'],
                ['Date & Time', violation.timestamp],
                ['Location', violation.location],
                ['Fine Amount', f'₹{violation.fine_amount}'],
                ['Applicable Rule', violation.rule],
                ['Detection Confidence', f'{violation.confidence:.1%}'],
                ['Status', violation.status.replace('_', ' ').title()],
            ]

            detail_table = Table(details_data, colWidths=[180, 310])
            detail_table.setStyle(TableStyle([
                # Header row
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a237e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                # Data rows
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f5f5f5')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')]),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                # Highlight fine amount row
                ('TEXTCOLOR', (1, 6), (1, 6), HexColor('#c62828')),
                ('FONTNAME', (1, 6), (1, 6), 'Helvetica-Bold'),
            ]))
            elements.append(detail_table)
            elements.append(Spacer(1, 16))

            # === SNAPSHOT IMAGE ===
            if violation.snapshot_path and os.path.exists(violation.snapshot_path):
                elements.append(Paragraph(
                    "<b>Violation Evidence (CCTV Snapshot)</b>",
                    self.styles['Heading3']
                ))
                elements.append(Spacer(1, 6))
                try:
                    img = RLImage(
                        violation.snapshot_path,
                        width=14 * cm,
                        height=8 * cm,
                    )
                    elements.append(img)
                except Exception as e:
                    logger.warning(f"Failed to embed snapshot: {e}")
                    elements.append(Paragraph(
                        "[Snapshot image not available]",
                        self.styles['Normal']
                    ))
                elements.append(Spacer(1, 12))

            # === QR CODE ===
            qr_img = self._generate_qr(
                f"VIOLATION:{violation.id}|PLATE:{violation.plate}|"
                f"FINE:{violation.fine_amount}|DATE:{violation.timestamp}"
            )
            if qr_img:
                elements.append(Paragraph(
                    "<b>Scan QR Code for Verification</b>",
                    ParagraphStyle(
                        name='QRLabel',
                        parent=self.styles['Normal'],
                        alignment=TA_CENTER,
                        fontSize=9,
                    )
                ))
                # Center QR code
                qr_table = Table([[qr_img]], colWidths=[14 * cm])
                qr_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                ]))
                elements.append(qr_table)
                elements.append(Spacer(1, 12))

            # === PAYMENT INSTRUCTIONS ===
            elements.append(HRFlowable(
                width="100%", thickness=1, color=HexColor('#cccccc')
            ))
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(
                "<b>Payment Instructions:</b><br/>"
                "1. Visit your nearest Motor Vehicles Office or<br/>"
                "2. Pay online at https://parivahan.gov.in<br/>"
                f"3. Quote Reference Number: <b>{violation.id}</b><br/>"
                "4. Payment must be made within 30 days of this notice.",
                self.styles['Normal']
            ))
            elements.append(Spacer(1, 16))

            # === LEGAL FOOTER ===
            elements.append(HRFlowable(
                width="100%", thickness=1, color=HexColor('#999999')
            ))
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(
                "This is a system-generated notice from the Automated Traffic "
                "Enforcement System. No signature is required.<br/>"
                "For queries, contact: Kerala MVD Helpline - 1800-425-1530<br/>"
                f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
                self.styles['Footer']
            ))

            # Build PDF
            doc.build(elements)
            logger.info(f"PDF report generated: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return ""
