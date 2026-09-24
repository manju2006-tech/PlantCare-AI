"""
Generates a downloadable PDF report for a single plant disease prediction.

Layout-adjusted version:
- Keeps the existing PDF generation function and data fields.
- Uses a side-by-side leaf image + prediction summary.
- Keeps all six information sections.
- Uses compact, readable spacing to fit the report on one A4 page.
- Keeps the footer pinned near the bottom.
"""
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, FrameBreak,
    Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable,
)

PRIMARY_COLOR = colors.HexColor('#2E7D32')
ACCENT_COLOR = colors.HexColor('#66BB6A')
DANGER_COLOR = colors.HexColor('#C62828')
HEALTHY_COLOR = colors.HexColor('#2E7D32')
TEXT_COLOR = colors.HexColor('#212121')
MUTED_COLOR = colors.HexColor('#616161')
LIGHT_BG = colors.HexColor('#F1F8E9')

# --- Page geometry ---
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_LEFT = 0.55 * inch
MARGIN_RIGHT = 0.55 * inch
MARGIN_TOP = 0.38 * inch
MARGIN_BOTTOM = 0.24 * inch
FOOTER_HEIGHT = 0.52 * inch

CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
CONTENT_TOP_Y = MARGIN_BOTTOM + FOOTER_HEIGHT
CONTENT_HEIGHT = PAGE_HEIGHT - MARGIN_TOP - CONTENT_TOP_Y


def _build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='ReportTitle',
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=22,
        textColor=PRIMARY_COLOR,
        alignment=TA_CENTER,
        spaceAfter=2,
    ))

    styles.add(ParagraphStyle(
        name='ReportSubtitle',
        fontName='Helvetica',
        fontSize=8.5,
        leading=10,
        textColor=MUTED_COLOR,
        alignment=TA_CENTER,
        spaceAfter=5,
    ))

    styles.add(ParagraphStyle(
        name='SectionHeading',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=13,
        textColor=colors.white,
        backColor=PRIMARY_COLOR,
        leftIndent=3,
        spaceBefore=4,
        spaceAfter=0,
        borderPadding=(3, 4, 3, 4),
    ))

    styles.add(ParagraphStyle(
        name='BodyTextCustom',
        fontName='Helvetica',
        fontSize=10.2,
        leading=13.2,
        textColor=TEXT_COLOR,
        spaceBefore=0,
        spaceAfter=0,
    ))

    styles.add(ParagraphStyle(
        name='FooterNote',
        fontName='Helvetica-Oblique',
        fontSize=6.5,
        textColor=MUTED_COLOR,
        alignment=TA_CENTER,
        leading=8,
    ))

    return styles


def _result_summary_table(prediction_data, col_width_left, col_width_right):
    is_healthy = prediction_data.get('is_healthy', False)
    status_text = 'HEALTHY' if is_healthy else 'DISEASE DETECTED'
    status_color = HEALTHY_COLOR if is_healthy else DANGER_COLOR

    rows = [
        ['Plant:', prediction_data.get('plant_name', 'Unknown')],
        ['Diagnosis:', prediction_data.get('disease_name', 'Unknown')],
        ['Confidence:', f"{prediction_data.get('confidence', 0):.2f}%"],
        ['Status:', status_text],
        ['Scan Date:', prediction_data.get(
            'created_at',
            datetime.utcnow().strftime('%Y-%m-%d %H:%M')
        )],
    ]

    table = Table(
        rows,
        colWidths=[col_width_left, col_width_right],
        hAlign='LEFT'
    )

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
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    return table


def _section(title, body_text, styles, content_width):
    heading = Paragraph(title, styles['SectionHeading'])
    body = Paragraph(body_text or 'Not available.', styles['BodyTextCustom'])

    # Give every section a substantial minimum height so the one-page report
    # uses the page instead of leaving a large empty area at the bottom.
    text_height = body.wrap(content_width - 24, 1000)[1]
    card_height = max(62, text_height + 22)

    card = Table(
        [[body]],
        colWidths=[content_width],
        rowHeights=[card_height],
    )
    card.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 0.75, ACCENT_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    return [heading, card]


def generate_prediction_report(prediction_data, user_data, image_path, report_folder):
    """
    Builds the PDF and writes it to `report_folder`.

    Returns the generated PDF's filename (not full path).
    """
    os.makedirs(report_folder, exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = f"plant_report_{timestamp}.pdf"
    output_path = os.path.join(report_folder, filename)

    styles = _build_styles()

    # Two-frame A4 layout: main content + fixed footer.
    content_frame = Frame(
        MARGIN_LEFT,
        CONTENT_TOP_Y,
        CONTENT_WIDTH,
        CONTENT_HEIGHT,
        id='content',
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )

    footer_frame = Frame(
        MARGIN_LEFT,
        MARGIN_BOTTOM,
        CONTENT_WIDTH,
        FOOTER_HEIGHT,
        id='footer',
        leftPadding=0,
        rightPadding=0,
        topPadding=2,
        bottomPadding=0,
    )

    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        title='Plant Disease Detection Report',
    )

    doc.addPageTemplates([
        PageTemplate(
            id='main',
            frames=[content_frame, footer_frame]
        )
    ])

    story = []

    # --- Header ---
    story.append(
        Paragraph(
            'Plant Disease Detection Report',
            styles['ReportTitle']
        )
    )

    story.append(
        Paragraph(
            f"AI-Powered Diagnosis &nbsp;|&nbsp; Generated on "
            f"{datetime.utcnow().strftime('%B %d, %Y at %H:%M UTC')}",
            styles['ReportSubtitle']
        )
    )

    story.append(
        HRFlowable(
            width='100%',
            thickness=1.1,
            color=ACCENT_COLOR
        )
    )

    story.append(Spacer(1, 8))

    # --- Side-by-side image + summary ---
    image_col_width = 2.45 * inch
    table_col_width = CONTENT_WIDTH - image_col_width

    summary_table = _result_summary_table(
        prediction_data,
        col_width_left=1.25 * inch,
        col_width_right=table_col_width - 1.25 * inch,
    )

    if image_path and os.path.exists(image_path):
        try:
            img = RLImage(
                image_path,
                width=2.25 * inch,
                height=2.25 * inch,
                kind='proportional'
            )
            img.hAlign = 'CENTER'

            top_section_table = Table(
                [[img, summary_table]],
                colWidths=[image_col_width, table_col_width]
            )

            top_section_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ]))

            story.append(top_section_table)

        except Exception:
            story.append(summary_table)
    else:
        story.append(summary_table)

    story.append(Spacer(1, 8))

    # --- Detailed sections ---
    story.extend(_section(
        'Description',
        prediction_data.get('description'),
        styles,
        CONTENT_WIDTH
    ))

    story.extend(_section(
        'Causes',
        prediction_data.get('causes'),
        styles,
        CONTENT_WIDTH
    ))

    story.extend(_section(
        'Treatment',
        prediction_data.get('treatment'),
        styles,
        CONTENT_WIDTH
    ))

    story.extend(_section(
        'Prevention',
        prediction_data.get('prevention'),
        styles,
        CONTENT_WIDTH
    ))

    story.extend(_section(
        'Fertilizer Recommendation',
        prediction_data.get('fertilizer_recommendation'),
        styles,
        CONTENT_WIDTH
    ))

    story.extend(_section(
        'Pesticide Recommendation',
        prediction_data.get('pesticide_recommendation'),
        styles,
        CONTENT_WIDTH
    ))

    # --- Fixed footer ---
    story.append(FrameBreak())

    story.append(
        HRFlowable(
            width='100%',
            thickness=0.5,
            color=colors.HexColor('#BDBDBD')
        )
    )

    story.append(Spacer(1, 3))

    story.append(
        Paragraph(
            'This report was generated by an AI model and is intended as a '
            'preliminary diagnostic aid only. For confirmed diagnosis and '
            'treatment planning, please consult a certified agronomist or '
            'plant pathologist.',
            styles['FooterNote']
        )
    )

    story.append(
        Paragraph(
            'Plant Disease Detection System - AI Plant Health Assistant',
            styles['FooterNote']
        )
    )

    doc.build(story)
    return filename
