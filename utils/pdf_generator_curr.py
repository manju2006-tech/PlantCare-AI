"""
Generates a downloadable PDF report for a single plant disease prediction,
combining the uploaded leaf image with the full diagnosis: disease name,
confidence, description, causes, treatment, prevention, and fertilizer /
pesticide recommendations.

Kept decoupled from the ORM (accepts plain dicts) so it can be unit tested
or reused outside a Flask request context.
"""
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable,
)

PRIMARY_COLOR = colors.HexColor('#2E7D32')   # deep green
ACCENT_COLOR = colors.HexColor('#66BB6A')    # light green
DANGER_COLOR = colors.HexColor('#C62828')    # red, used for diseased status
HEALTHY_COLOR = colors.HexColor('#2E7D32')   # green, used for healthy status
TEXT_COLOR = colors.HexColor('#212121')
MUTED_COLOR = colors.HexColor('#616161')
LIGHT_BG = colors.HexColor('#F1F8E9')


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ReportTitle', fontName='Helvetica-Bold', fontSize=17,
        textColor=PRIMARY_COLOR, alignment=TA_CENTER, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name='ReportSubtitle', fontName='Helvetica', fontSize=8.5,
        textColor=MUTED_COLOR, alignment=TA_CENTER, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name='SectionHeading', fontName='Helvetica-Bold', fontSize=9.5,
        textColor=colors.white, backColor=PRIMARY_COLOR,
        leftIndent=6, spaceBefore=7, spaceAfter=3,
        borderPadding=(3, 4, 3, 4),
    ))
    styles.add(ParagraphStyle(
        name='BodyTextCustom', fontName='Helvetica', fontSize=8.5,
        textColor=TEXT_COLOR, leading=10.5, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name='FooterNote', fontName='Helvetica-Oblique', fontSize=7,
        textColor=MUTED_COLOR, alignment=TA_CENTER, leading=9,
    ))
    return styles


def _result_summary_table(prediction_data, styles):
    is_healthy = prediction_data.get('is_healthy', False)
    status_text = 'HEALTHY' if is_healthy else 'DISEASE DETECTED'
    status_color = HEALTHY_COLOR if is_healthy else DANGER_COLOR

    rows = [
        ['Plant:', prediction_data.get('plant_name', 'Unknown')],
        ['Diagnosis:', prediction_data.get('disease_name', 'Unknown')],
        ['Confidence:', f"{prediction_data.get('confidence', 0):.2f}%"],
        ['Status:', status_text],
        ['Scan Date:', prediction_data.get('created_at', datetime.utcnow().strftime('%Y-%m-%d %H:%M'))],
    ]

    table = Table(rows, colWidths=[1.6 * inch, 4.4 * inch], hAlign='LEFT')
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), MUTED_COLOR),
        ('TEXTCOLOR', (1, 0), (1, -1), TEXT_COLOR),
        ('TEXTCOLOR', (1, 3), (1, 3), status_color),
        ('FONTNAME', (1, 3), (1, 3), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 0.75, ACCENT_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.white),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    return table


def _section(title, body_text, styles):
    return [
        Paragraph(title, styles['SectionHeading']),
        Paragraph(body_text or 'Not available.', styles['BodyTextCustom']),
    ]


def generate_prediction_report(prediction_data, user_data, image_path, report_folder):
    """
    Builds the PDF and writes it to `report_folder`.

    Args:
        prediction_data: dict with keys disease_name, plant_name, is_healthy,
            confidence, description, causes, treatment, prevention,
            fertilizer_recommendation, pesticide_recommendation, created_at.
        user_data: dict with keys username, email, full_name.
        image_path: absolute path to the uploaded leaf image (may be None
            or missing on disk - handled gracefully).
        report_folder: directory to save the PDF into.

    Returns:
        The generated PDF's filename (not full path) - store this in the
        Prediction.report_filename column.
    """
    os.makedirs(report_folder, exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = f"plant_report_{timestamp}.pdf"
    output_path = os.path.join(report_folder, filename)

    styles = _build_styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=0.4 * inch, bottomMargin=0.4 * inch,
        leftMargin=0.55 * inch, rightMargin=0.55 * inch,
        title='Plant Disease Detection Report',
    )

    story = []

    # --- Header ---
    story.append(Paragraph('Plant Disease Detection Report', styles['ReportTitle']))
    story.append(Paragraph(
        f"AI-Powered Diagnosis &nbsp;|&nbsp; Generated on "
        f"{datetime.utcnow().strftime('%B %d, %Y at %H:%M UTC')}",
        styles['ReportSubtitle']
    ))
    story.append(HRFlowable(width='100%', thickness=1, color=ACCENT_COLOR))
    story.append(Spacer(1, 6))

    # --- Scanned-by info ---
    scanned_by = user_data.get('full_name') or user_data.get('username', 'Unknown User')
    story.append(Paragraph(
        f"<b>Scanned by:</b> {scanned_by} &nbsp;(&nbsp;{user_data.get('email', '')}&nbsp;)",
        styles['BodyTextCustom']
    ))
    story.append(Spacer(1, 4))

    # --- Uploaded image (if it exists on disk) ---
    if image_path and os.path.exists(image_path):
        try:
            img = RLImage(image_path, width=1.5 * inch, height=1.5 * inch, kind='proportional')
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 5))
        except Exception:
            pass  # if the image can't be embedded, still generate the rest of the report

    # --- Result summary table ---
    story.append(_result_summary_table(prediction_data, styles))
    story.append(Spacer(1, 6))

    # --- Detailed sections ---
    story.extend(_section('Description', prediction_data.get('description'), styles))
    story.extend(_section('Causes', prediction_data.get('causes'), styles))
    story.extend(_section('Treatment', prediction_data.get('treatment'), styles))
    story.extend(_section('Prevention', prediction_data.get('prevention'), styles))
    story.extend(_section(
        'Fertilizer Recommendation', prediction_data.get('fertilizer_recommendation'), styles
    ))
    story.extend(_section(
        'Pesticide Recommendation', prediction_data.get('pesticide_recommendation'), styles
    ))

    # --- Footer disclaimer ---
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#BDBDBD')))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'This report was generated by an AI model and is intended as a preliminary '
        'diagnostic aid only. For confirmed diagnosis and treatment planning, please '
        'consult a certified agronomist or plant pathologist.',
        styles['FooterNote']
    ))
    story.append(Paragraph('Plant Disease Detection System - AI Plant Health Assistant', styles['FooterNote']))

    doc.build(story)
    return filename