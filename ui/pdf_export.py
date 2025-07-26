"""
PDF Export utilities for AI Visibility Audit Tool - Startup Tech Style
"""
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether, \
    HRFlowable, Frame, PageTemplate, BaseDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Polygon
from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics import renderPDF
from datetime import datetime
import io
import base64
from typing import Dict, Any, List
import os
from translations import get_text
import math

# Keyweo brand colors - Startup Tech palette
KEYWEO_GREEN = colors.HexColor('#00D9A3')
KEYWEO_DARK = colors.HexColor('#0A0E27')
KEYWEO_PURPLE = colors.HexColor('#6366F1')
KEYWEO_PINK = colors.HexColor('#EC4899')
KEYWEO_BLUE = colors.HexColor('#3B82F6')
KEYWEO_GRAY = colors.HexColor('#64748B')
KEYWEO_LIGHT_GRAY = colors.HexColor('#F1F5F9')
KEYWEO_WHITE = colors.HexColor('#FFFFFF')

# Gradient colors for charts
GRADIENT_COLORS = [
    colors.HexColor('#00D9A3'),
    colors.HexColor('#00C896'),
    colors.HexColor('#00B789'),
    colors.HexColor('#00A67C'),
    colors.HexColor('#009570'),
]


def create_gradient_rect(x, y, width, height, color1, color2, vertical=True):
    """Create a gradient rectangle"""
    drawing = Drawing(width, height)

    # Number of steps for gradient
    steps = 20
    if vertical:
        step_height = height / steps
        for i in range(steps):
            # Interpolate color
            ratio = i / steps
            r = int(color1.red * 255 * (1 - ratio) + color2.red * 255 * ratio)
            g = int(color1.green * 255 * (1 - ratio) + color2.green * 255 * ratio)
            b = int(color1.blue * 255 * (1 - ratio) + color2.blue * 255 * ratio)

            rect = Rect(0, i * step_height, width, step_height,
                        fillColor=colors.Color(r / 255, g / 255, b / 255),
                        strokeColor=None)
            drawing.add(rect)

    return drawing


def create_cover_page(brand_name: str, brand_url: str) -> List:
    """Create a modern startup-style cover page"""
    elements = []

    # Background gradient effect
    gradient = create_gradient_rect(0, 0, A4[0], A4[1], KEYWEO_DARK, KEYWEO_PURPLE)

    # Large Keyweo branding
    elements.append(Spacer(1, 2 * inch))

    # Logo area with animated-style elements
    logo_style = ParagraphStyle(
        'Logo',
        fontSize=48,
        textColor=KEYWEO_WHITE,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=56
    )

    elements.append(Paragraph('<font color="#00D9A3">key</font>weo', logo_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Tagline
    tagline_style = ParagraphStyle(
        'Tagline',
        fontSize=16,
        textColor=KEYWEO_WHITE,
        alignment=TA_CENTER,
        fontName='Helvetica',
        opacity=0.8
    )
    elements.append(Paragraph('AI Visibility Intelligence Platform', tagline_style))

    elements.append(Spacer(1, 2 * inch))

    # Report title with modern styling
    title_style = ParagraphStyle(
        'CoverTitle',
        fontSize=36,
        textColor=KEYWEO_WHITE,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=44
    )

    elements.append(Paragraph('AI VISIBILITY', title_style))
    elements.append(Paragraph('AUDIT REPORT', title_style))

    elements.append(Spacer(1, 1 * inch))

    # Brand info in a modern card style
    brand_style = ParagraphStyle(
        'BrandInfo',
        fontSize=24,
        textColor=KEYWEO_GREEN,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    elements.append(Paragraph(brand_name.upper(), brand_style))

    if brand_url:
        url_style = ParagraphStyle(
            'URLInfo',
            fontSize=14,
            textColor=KEYWEO_WHITE,
            alignment=TA_CENTER,
            fontName='Helvetica',
            opacity=0.7
        )
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(brand_url, url_style))

    # Date at bottom
    elements.append(Spacer(1, 3 * inch))
    date_style = ParagraphStyle(
        'DateStyle',
        fontSize=12,
        textColor=KEYWEO_WHITE,
        alignment=TA_CENTER,
        fontName='Helvetica',
        opacity=0.6
    )
    elements.append(Paragraph(datetime.now().strftime('%B %Y').upper(), date_style))

    return elements


def create_executive_dashboard(results: Dict[str, Any], brand_name: str,
                               prompts: List[str], competitors: List[str]) -> List:
    """Create an executive dashboard page"""
    elements = []

    # Page header
    header_style = ParagraphStyle(
        'DashHeader',
        fontSize=28,
        textColor=KEYWEO_DARK,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold',
        spaceAfter=20
    )

    elements.append(Paragraph('Executive Dashboard', header_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=KEYWEO_GREEN))
    elements.append(Spacer(1, 0.3 * inch))

    # Calculate metrics
    total_unique = sum(results.get(p, {}).get('unique_mentions', 0) for p in results.keys())
    total_mentions = sum(results.get(p, {}).get('total_mentions', 0) for p in results.keys())
    visibility_rate = (total_unique / (len(prompts) * len(results))) * 100 if results else 0

    # Create modern metric cards
    metrics_data = []

    # Row 1: Main metrics
    metric_card_style = '''
        <para alignment="center">
            <font size="36" color="{color}"><b>{value}</b></font><br/>
            <font size="11" color="#64748B">{label}</font>
        </para>
    '''

    metrics_row1 = [
        Paragraph(metric_card_style.format(
            color=KEYWEO_PURPLE.hexval(),
            value=len(prompts),
            label='Queries Tested'
        ), ParagraphStyle('Metric')),

        Paragraph(metric_card_style.format(
            color=KEYWEO_GREEN.hexval(),
            value=total_unique,
            label='Unique Mentions'
        ), ParagraphStyle('Metric')),

        Paragraph(metric_card_style.format(
            color=KEYWEO_BLUE.hexval(),
            value=total_mentions,
            label='Total Mentions'
        ), ParagraphStyle('Metric')),

        Paragraph(metric_card_style.format(
            color=KEYWEO_PINK.hexval(),
            value=f'{visibility_rate:.0f}%',
            label='Visibility Score'
        ), ParagraphStyle('Metric'))
    ]

    # Create metric table
    metrics_table = Table([metrics_row1], colWidths=[1.75 * inch] * 4)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), KEYWEO_LIGHT_GRAY),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('TOPPADDING', (0, 0), (-1, -1), 20),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
        ('GRID', (0, 0), (-1, -1), 2, KEYWEO_WHITE),
        ('ROUNDEDCORNERS', [5, 5, 5, 5]),
    ]))

    elements.append(metrics_table)
    elements.append(Spacer(1, 0.4 * inch))

    # Platform performance mini chart
    elements.append(Paragraph('Platform Performance', subheading_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Create platform performance visualization
    platform_chart = create_platform_performance_chart(results, len(prompts))
    elements.append(platform_chart)

    # Competitive landscape
    if competitors:
        elements.append(Spacer(1, 0.4 * inch))
        elements.append(Paragraph('Competitive Landscape', subheading_style))
        elements.append(Spacer(1, 0.2 * inch))

        # Create competitive positioning chart
        comp_chart = create_competitive_positioning(results, brand_name, competitors, prompts)
        elements.append(comp_chart)

    return elements


def create_platform_performance_chart(results: Dict[str, Any], num_prompts: int) -> Drawing:
    """Create a modern platform performance visualization"""
    drawing = Drawing(500, 200)

    # Data preparation
    platforms = []
    unique_mentions = []
    total_mentions = []
    visibility_rates = []

    platform_display = {
        'chatgpt': 'ChatGPT',
        'perplexity': 'Perplexity',
        'gemini': 'Gemini'
    }

    for platform in ['chatgpt', 'perplexity', 'gemini']:
        if platform in results:
            platforms.append(platform_display[platform])
            unique = results[platform].get('unique_mentions', 0)
            total = results[platform].get('total_mentions', 0)
            rate = (unique / num_prompts * 100) if num_prompts > 0 else 0

            unique_mentions.append(unique)
            total_mentions.append(total)
            visibility_rates.append(rate)

    # Create modern bar chart
    bc = HorizontalBarChart()
    bc.x = 100
    bc.y = 50
    bc.height = 120
    bc.width = 350

    # Data for stacked bars
    bc.data = [unique_mentions, [t - u for t, u in zip(total_mentions, unique_mentions)]]
    bc.categoryAxis.categoryNames = platforms

    # Styling
    bc.bars[0].fillColor = KEYWEO_GREEN
    bc.bars[1].fillColor = KEYWEO_GREEN
    bc.bars[1].fillColor.alpha = 0.3

    bc.categoryAxis.labels.fontName = 'Helvetica'
    bc.categoryAxis.labels.fontSize = 11
    bc.valueAxis.labels.fontName = 'Helvetica'
    bc.valueAxis.labels.fontSize = 9

    # Add percentage labels
    for i, (platform, rate) in enumerate(zip(platforms, visibility_rates)):
        y_pos = bc.y + bc.height - (i + 0.5) * (bc.height / len(platforms))
        drawing.add(String(460, y_pos - 5, f'{rate:.0f}%',
                           fontSize=11, fillColor=KEYWEO_PURPLE,
                           fontName='Helvetica-Bold'))

    drawing.add(bc)

    # Legend
    legend_items = [
        ('Unique Mentions', KEYWEO_GREEN),
        ('Repeat Mentions', KEYWEO_GREEN, 0.3)
    ]

    legend_y = 180
    for i, item in enumerate(legend_items):
        x = 150 + i * 120
        color = item[1]
        if len(item) > 2:
            color = colors.Color(color.red, color.green, color.blue, alpha=item[2])

        drawing.add(Rect(x, legend_y, 15, 10, fillColor=color, strokeColor=None))
        drawing.add(String(x + 20, legend_y, item[0], fontSize=9))

    return drawing


def create_competitive_positioning(results: Dict[str, Any], brand_name: str,
                                   competitors: List[str], prompts: List[str]) -> Drawing:
    """Create a competitive positioning bubble chart"""
    drawing = Drawing(500, 250)

    # Calculate positioning data
    brand_data = []

    # Add main brand
    brand_unique = sum(results[p].get('unique_mentions', 0) for p in results.keys())
    brand_total = sum(results[p].get('total_mentions', 0) for p in results.keys())
    brand_rate = (brand_unique / (len(prompts) * len(results))) * 100 if results else 0

    brand_data.append({
        'name': brand_name,
        'unique': brand_unique,
        'total': brand_total,
        'rate': brand_rate,
        'is_main': True
    })

    # Add competitors
    for comp in competitors[:10]:
        comp_unique = 0
        comp_total = 0
        for platform in results.keys():
            for response in results[platform].get('responses', []):
                if response.get('competitor_mentions', {}).get(comp, 0) > 0:
                    comp_unique += 1
                    comp_total += response.get('competitor_mentions', {}).get(comp, 0)

        comp_rate = (comp_unique / (len(prompts) * len(results))) * 100 if results else 0

        if comp_unique > 0:
            brand_data.append({
                'name': comp,
                'unique': comp_unique,
                'total': comp_total,
                'rate': comp_rate,
                'is_main': False
            })

    # Sort by rate
    brand_data.sort(key=lambda x: x['rate'], reverse=True)

    # Create visualization
    max_rate = max(b['rate'] for b in brand_data) if brand_data else 100

    y_pos = 200
    for i, brand in enumerate(brand_data[:8]):  # Top 8 brands
        # Bar
        bar_width = (brand['rate'] / max_rate) * 300 if max_rate > 0 else 0
        color = KEYWEO_GREEN if brand['is_main'] else KEYWEO_GRAY

        drawing.add(Rect(120, y_pos - i * 25, bar_width, 18,
                         fillColor=color, strokeColor=None))

        # Brand name
        name = brand['name']
        if len(name) > 15:
            name = name[:12] + '...'

        text_color = KEYWEO_DARK if brand['is_main'] else KEYWEO_GRAY
        drawing.add(String(10, y_pos - i * 25 + 5, name,
                           fontSize=10, fillColor=text_color,
                           fontName='Helvetica-Bold' if brand['is_main'] else 'Helvetica'))

        # Value
        drawing.add(String(125 + bar_width, y_pos - i * 25 + 5, f"{brand['rate']:.1f}%",
                           fontSize=9, fillColor=KEYWEO_DARK))

    return drawing


def create_detailed_analysis_pages(results: Dict[str, Any], prompts: List[str],
                                   brand_name: str, competitors: List[str]) -> List:
    """Create detailed analysis pages for each prompt"""
    elements = []

    for prompt_idx, prompt in enumerate(prompts):
        # New page for each prompt
        if prompt_idx > 0:
            elements.append(PageBreak())

        # Prompt header
        elements.append(Paragraph(f'Query Analysis #{prompt_idx + 1}', heading_style))
        elements.append(HRFlowable(width="100%", thickness=1, color=KEYWEO_GREEN))
        elements.append(Spacer(1, 0.2 * inch))

        # Prompt text
        prompt_style = ParagraphStyle(
            'PromptText',
            fontSize=14,
            textColor=KEYWEO_DARK,
            alignment=TA_LEFT,
            fontName='Helvetica',
            leftIndent=20,
            rightIndent=20,
            spaceAfter=20,
            borderColor=KEYWEO_LIGHT_GRAY,
            borderWidth=1,
            borderPadding=10,
            backColor=KEYWEO_LIGHT_GRAY
        )

        elements.append(Paragraph(f'<i>"{prompt}"</i>', prompt_style))
        elements.append(Spacer(1, 0.3 * inch))

        # Create modern heatmap
        heatmap = create_modern_heatmap(results, prompts, brand_name, competitors, prompt_idx)
        elements.append(heatmap)
        elements.append(Spacer(1, 0.3 * inch))

        # Platform responses analysis
        elements.append(Paragraph('Platform Responses', subheading_style))
        elements.append(Spacer(1, 0.1 * inch))

        # Create response cards for each platform
        for platform in results.keys():
            if prompt_idx < len(results[platform]['responses']):
                response_data = results[platform]['responses'][prompt_idx]

                # Platform card
                card_elements = []

                # Platform name and status
                platform_name = platform.title()
                is_mentioned = response_data.get('mentions', 0) > 0
                status_color = KEYWEO_GREEN if is_mentioned else KEYWEO_GRAY
                status_text = "✓ Found" if is_mentioned else "✗ Not found"

                card_header = Table([
                    [Paragraph(f'<b>{platform_name}</b>',
                               ParagraphStyle('PlatformName', fontSize=12, textColor=KEYWEO_DARK)),
                     Paragraph(status_text,
                               ParagraphStyle('Status', fontSize=10, textColor=status_color, alignment=TA_RIGHT))]
                ], colWidths=[3 * inch, 1.5 * inch])

                card_header.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))

                card_elements.append(card_header)

                # Extract key mention
                if is_mentioned:
                    response_text = response_data.get('response', '')
                    key_extract = extract_key_mention(response_text, brand_name)

                    extract_style = ParagraphStyle(
                        'Extract',
                        fontSize=9,
                        textColor=KEYWEO_GRAY,
                        leftIndent=10,
                        rightIndent=10,
                        spaceAfter=5
                    )

                    card_elements.append(Paragraph(key_extract, extract_style))

                # Competitors found
                comp_found = []
                for comp in competitors:
                    if response_data.get('competitor_mentions', {}).get(comp, 0) > 0:
                        comp_found.append(comp)

                if comp_found:
                    comp_text = f"Competitors: {', '.join(comp_found[:5])}"
                    if len(comp_found) > 5:
                        comp_text += f" +{len(comp_found) - 5} more"

                    card_elements.append(Paragraph(comp_text,
                                                   ParagraphStyle('Competitors', fontSize=8,
                                                                  textColor=KEYWEO_GRAY, italic=True)))

                # Create card table
                card = Table([card_elements], colWidths=[4.5 * inch])
                card.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), KEYWEO_WHITE),
                    ('BOX', (0, 0), (-1, -1), 1, KEYWEO_LIGHT_GRAY),
                    ('LEFTPADDING', (0, 0), (-1, -1), 10),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ]))

                elements.append(card)
                elements.append(Spacer(1, 0.15 * inch))

    return elements


def create_modern_heatmap(results: Dict[str, Any], prompts: List[str],
                          brand_name: str, competitors: List[str], prompt_idx: int) -> Drawing:
    """Create a modern, clean heatmap visualization"""
    drawing = Drawing(500, 200)

    # Get relevant competitors for this prompt
    relevant_competitors = []
    platforms = list(results.keys())

    for comp in competitors:
        for platform in platforms:
            if prompt_idx < len(results[platform]['responses']):
                response = results[platform]['responses'][prompt_idx]
                if response.get('competitor_mentions', {}).get(comp, 0) > 0:
                    relevant_competitors.append(comp)
                    break

    # Limit brands shown
    all_brands = [brand_name] + relevant_competitors[:7]

    # Modern cell design
    cell_size = 40
    gap = 3
    start_x = 100
    start_y = 150

    # Draw cells
    for i, platform in enumerate(platforms):
        # Platform label
        platform_label = platform.title()
        drawing.add(String(start_x - 10, start_y - i * (cell_size + gap) - cell_size / 2 + 5,
                           platform_label, fontSize=11, textAnchor='end',
                           fillColor=KEYWEO_DARK, fontName='Helvetica'))

        if prompt_idx < len(results[platform]['responses']):
            response = results[platform]['responses'][prompt_idx]

            for j, brand in enumerate(all_brands):
                x = start_x + j * (cell_size + gap)
                y = start_y - i * (cell_size + gap) - cell_size

                # Determine presence and color
                if j == 0:  # Main brand
                    present = response.get('mentions', 0) > 0
                    cell_color = KEYWEO_GREEN if present else KEYWEO_LIGHT_GRAY
                else:  # Competitor
                    present = response.get('competitor_mentions', {}).get(brand, 0) > 0
                    cell_color = KEYWEO_PURPLE if present else KEYWEO_LIGHT_GRAY

                # Draw rounded rectangle cell
                drawing.add(Rect(x, y, cell_size, cell_size,
                                 fillColor=cell_color,
                                 strokeColor=KEYWEO_WHITE,
                                 strokeWidth=2,
                                 rx=5, ry=5))

                # Add checkmark or X
                if present:
                    drawing.add(String(x + cell_size / 2, y + cell_size / 2 - 5, '✓',
                                       fontSize=16, textAnchor='middle',
                                       fillColor=KEYWEO_WHITE, fontName='Helvetica-Bold'))

    # Brand labels with better positioning
    for j, brand in enumerate(all_brands):
        x = start_x + j * (cell_size + gap) + cell_size / 2
        y = start_y - len(platforms) * (cell_size + gap) - 15

        # Truncate and style
        label = brand if len(brand) <= 10 else brand[:8] + '..'
        label_color = KEYWEO_GREEN if j == 0 else KEYWEO_DARK

        drawing.add(String(x, y, label, fontSize=9, textAnchor='middle',
                           fillColor=label_color, fontName='Helvetica'))

    return drawing


def extract_key_mention(text: str, brand_name: str) -> str:
    """Extract key mention with context"""
    import re

    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        if brand_name.lower() in sent.lower():
            sent_clean = sent.strip()

            # Highlight brand name
            highlighted = sent_clean.replace(brand_name, f'<b>{brand_name}</b>')

            # Limit length
            if len(highlighted) > 150:
                brand_pos = highlighted.lower().find(brand_name.lower())
                start = max(0, brand_pos - 50)
                end = min(len(highlighted), brand_pos + 100)

                if start > 0:
                    highlighted = '...' + highlighted[start:end]
                else:
                    highlighted = highlighted[:end]

                if end < len(sent_clean):
                    highlighted += '...'

            return highlighted

    return f'<b>{brand_name}</b> mentioned in response'


def create_data_tables_section(results: Dict[str, Any], brand_name: str,
                               competitors: List[str], prompts: List[str]) -> List:
    """Create detailed data tables section"""
    elements = []

    elements.append(PageBreak())
    elements.append(Paragraph('Data Tables', heading_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=KEYWEO_GREEN))
    elements.append(Spacer(1, 0.3 * inch))

    # Platform comparison table
    elements.append(Paragraph('Platform Comparison Matrix', subheading_style))
    elements.append(Spacer(1, 0.1 * inch))

    platform_data = [
        ['Platform', 'Unique\nMentions', 'Total\nMentions', 'Visibility\nRate', 'Avg. Mentions\nper Query']]

    for platform in results.keys():
        unique = results[platform].get('unique_mentions', 0)
        total = results[platform].get('total_mentions', 0)
        rate = (unique / len(prompts) * 100) if len(prompts) > 0 else 0
        avg = total / len(prompts) if len(prompts) > 0 else 0

        platform_data.append([
            platform.title(),
            str(unique),
            str(total),
            f'{rate:.1f}%',
            f'{avg:.1f}'
        ])

    platform_table = Table(platform_data, colWidths=[1.5 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.4 * inch])
    platform_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), KEYWEO_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), KEYWEO_WHITE),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), KEYWEO_WHITE),
        ('GRID', (0, 0), (-1, -1), 1, KEYWEO_LIGHT_GRAY),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [KEYWEO_WHITE, KEYWEO_LIGHT_GRAY]),
    ]))

    elements.append(platform_table)
    elements.append(Spacer(1, 0.4 * inch))

    # Query-level breakdown
    elements.append(Paragraph('Query-Level Breakdown', subheading_style))
    elements.append(Spacer(1, 0.1 * inch))

    query_data = [['Query', 'ChatGPT', 'Perplexity', 'Gemini', 'Total']]

    for i, prompt in enumerate(prompts):
        row = [f'Q{i + 1}']
        total_for_query = 0

        for platform in ['chatgpt', 'perplexity', 'gemini']:
            if platform in results and i < len(results[platform]['responses']):
                mentions = results[platform]['responses'][i].get('mentions', 0)
                row.append('✓' if mentions > 0 else '-')
                total_for_query += 1 if mentions > 0 else 0
            else:
                row.append('-')

        row.append(f'{total_for_query}/3')
        query_data.append(row)

    query_table = Table(query_data, colWidths=[0.8 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch, 1 * inch])
    query_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), KEYWEO_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), KEYWEO_WHITE),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), KEYWEO_WHITE),
        ('GRID', (0, 0), (-1, -1), 1, KEYWEO_LIGHT_GRAY),
        ('TEXTCOLOR', (1, 1), (3, -1), KEYWEO_GREEN),
        ('FONTNAME', (1, 1), (3, -1), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [KEYWEO_WHITE, KEYWEO_LIGHT_GRAY]),
    ]))

    elements.append(query_table)

    return elements


def add_modern_header_footer(canvas_obj, doc):
    """Add modern header and footer"""
    canvas_obj.saveState()

    # Header - minimal and modern
    if doc.page > 1:  # Skip header on cover page
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(KEYWEO_GRAY)
        canvas_obj.drawRightString(A4[0] - 56, A4[1] - 40, f'Page {doc.page}')

        # Thin line
        canvas_obj.setStrokeColor(KEYWEO_LIGHT_GRAY)
        canvas_obj.setLineWidth(1)
        canvas_obj.line(56, A4[1] - 50, A4[0] - 56, A4[1] - 50)

    # Footer - modern branding
    if doc.page > 1:
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(KEYWEO_GRAY)
        canvas_obj.drawString(56, 40, 'AI Visibility Audit Report')
        canvas_obj.drawRightString(A4[0] - 56, 40, 'Powered by Keyweo')

        # Footer line
        canvas_obj.setStrokeColor(KEYWEO_GREEN)
        canvas_obj.setLineWidth(2)
        canvas_obj.line(56, 50, A4[0] - 56, 50)

    canvas_obj.restoreState()


# Global styles
heading_style = ParagraphStyle(
    'ModernHeading',
    fontSize=20,
    textColor=KEYWEO_DARK,
    spaceAfter=12,
    fontName='Helvetica-Bold'
)

subheading_style = ParagraphStyle(
    'ModernSubheading',
    fontSize=14,
    textColor=KEYWEO_DARK,
    spaceAfter=8,
    fontName='Helvetica-Bold'
)

normal_style = ParagraphStyle(
    'ModernNormal',
    fontSize=10,
    textColor=KEYWEO_GRAY,
    alignment=TA_LEFT,
    spaceAfter=8
)


def generate_pdf_report(
        brand_name: str,
        brand_url: str,
        results: Dict[str, Any],
        competitors: List[str],
        prompts: List[str],
        lang: str = 'en'
) -> bytes:
    """Generate a complete PDF report in startup tech style"""

    # Create PDF buffer
    buffer = io.BytesIO()

    # Create document with custom margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=56,
        leftMargin=56,
        topMargin=72,
        bottomMargin=72
    )

    # Container for all elements
    elements = []

    # 1. Cover page
    elements.extend(create_cover_page(brand_name, brand_url))
    elements.append(PageBreak())

    # 2. Executive dashboard
    elements.extend(create_executive_dashboard(results, brand_name, prompts, competitors))
    elements.append(PageBreak())

    # 3. Detailed analysis for each prompt
    elements.extend(create_detailed_analysis_pages(results, prompts, brand_name, competitors))

    # 4. Data tables section
    elements.extend(create_data_tables_section(results, brand_name, competitors, prompts))

    # Build PDF
    doc.build(elements, onFirstPage=add_modern_header_footer, onLaterPages=add_modern_header_footer)

    # Get PDF data
    pdf_data = buffer.getvalue()
    buffer.close()

    return pdf_data