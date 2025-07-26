"""
Competitor visualization components
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
from translations import get_text


def create_competitor_comparison_table(results: Dict[str, Any], brand_name: str,
                                       competitors: List[str]) -> pd.DataFrame:
    """Create a comparison table of brand vs competitors mentions with unique and total counts"""
    lang = st.session_state.get('language', 'en')

    # Initialize data structure
    all_brands = [brand_name] + competitors
    platforms = list(results.keys())

    # Create data for table
    data = []

    for brand in all_brands:
        row = {get_text('brand', lang): brand}

        # Initialize unique and total counters
        unique_mentions_total = 0
        total_mentions_sum = 0

        for platform in platforms:
            if brand == brand_name:
                # Get brand mentions (unique and total)
                unique_mentions = 0
                total_mentions = 0

                for response in results[platform]['responses']:
                    mentions_count = response['mentions']
                    if mentions_count > 0:
                        unique_mentions += 1  # Count as 1 if present
                        total_mentions += mentions_count  # Add actual count

                row[f'{platform.title()} (Unique)'] = unique_mentions
                row[f'{platform.title()} (Total)'] = total_mentions
                unique_mentions_total += unique_mentions
                total_mentions_sum += total_mentions
            else:
                # Get competitor mentions
                unique_mentions = 0
                total_mentions = 0

                for response in results[platform]['responses']:
                    comp_mentions = response.get('competitor_mentions', {}).get(brand, 0)
                    if comp_mentions > 0:
                        unique_mentions += 1  # Count as 1 if present
                        total_mentions += comp_mentions  # Add actual count

                row[f'{platform.title()} (Unique)'] = unique_mentions
                row[f'{platform.title()} (Total)'] = total_mentions
                unique_mentions_total += unique_mentions
                total_mentions_sum += total_mentions

        # Calculate totals
        row[get_text('unique_mentions', lang)] = unique_mentions_total
        row[get_text('total_mentions', lang)] = total_mentions_sum

        # Calculate visibility rate based on UNIQUE mentions
        total_prompts = len(results[platforms[0]]['responses']) * len(platforms)
        row[get_text('visibility_rate', lang) + ' %'] = round((unique_mentions_total / total_prompts) * 100, 1)

        data.append(row)

    # Create DataFrame and sort by unique mentions
    df = pd.DataFrame(data)
    df = df.sort_values(get_text('unique_mentions', lang), ascending=False)

    return df


def display_competitor_comparison_chart(df: pd.DataFrame, brand_name: str):
    """Display stacked bar chart comparing brand vs competitors using UNIQUE mentions"""
    lang = st.session_state.get('language', 'en')

    # Get unique mention columns only
    platforms = []
    for col in df.columns:
        if '(Unique)' in col:
            platform_name = col.replace(' (Unique)', '')
            platforms.append(platform_name)

    # Create stacked bar chart
    fig = go.Figure()

    # Add bars for each platform (using unique mentions)
    for platform in platforms:
        fig.add_trace(go.Bar(
            name=platform,
            x=df[get_text('brand', lang)],
            y=df[f'{platform} (Unique)'],
            text=df[f'{platform} (Unique)'],
            textposition='auto',
        ))

    # Update layout
    fig.update_layout(
        title=get_text('brand_presence_title', lang),
        xaxis_title=get_text('brand', lang) + 's',
        yaxis_title=get_text('prompts', lang),
        barmode='stack',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Highlight the main brand
    brand_col = get_text('brand', lang)
    for i, brand in enumerate(df[brand_col]):
        if brand == brand_name:
            fig.add_annotation(
                x=brand,
                y=df.loc[df[brand_col] == brand, get_text('unique_mentions', lang)].values[0] + 0.5,
                text=get_text('your_brand_presence', lang).split()[0],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#FF6B6B",
                ax=0,
                ay=-40
            )

    return fig


def display_total_mentions_chart(df: pd.DataFrame, brand_name: str):
    """Display chart showing total mentions (multiple per prompt)"""
    lang = st.session_state.get('language', 'en')

    # Get total mention columns
    platforms = []
    for col in df.columns:
        if '(Total)' in col:
            platform_name = col.replace(' (Total)', '')
            platforms.append(platform_name)

    # Create grouped bar chart
    fig = go.Figure()

    # Add bars for each platform
    for platform in platforms:
        fig.add_trace(go.Bar(
            name=platform,
            x=df[get_text('brand', lang)],
            y=df[f'{platform} (Total)'],
            text=df[f'{platform} (Total)'],
            textposition='auto',
        ))

    # Update layout
    fig.update_layout(
        title=get_text('total_mentions_title', lang),
        xaxis_title=get_text('brand', lang) + 's',
        yaxis_title=get_text('mentions', lang),
        barmode='group',
        height=500,
        showlegend=True
    )

    return fig


def display_visibility_rate_comparison(df: pd.DataFrame, brand_name: str):
    """Display horizontal bar chart for visibility rates (based on unique mentions)"""
    lang = st.session_state.get('language', 'en')

    # Sort by visibility rate
    vis_rate_col = get_text('visibility_rate', lang) + ' %'
    df_sorted = df.sort_values(vis_rate_col, ascending=True)

    # Create color list - highlight the main brand
    brand_col = get_text('brand', lang)
    colors = ['#FF6B6B' if brand == brand_name else '#4ECDC4' for brand in df_sorted[brand_col]]

    fig = go.Figure(go.Bar(
        x=df_sorted[vis_rate_col],
        y=df_sorted[brand_col],
        orientation='h',
        marker_color=colors,
        text=df_sorted[vis_rate_col].apply(lambda x: f'{x}%'),
        textposition='outside',
    ))

    fig.update_layout(
        title=get_text('visibility_rate_title', lang),
        xaxis_title=get_text('visibility_rate', lang) + " (%)",
        yaxis_title="",
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, max(df_sorted[vis_rate_col]) * 1.2])
    )

    return fig


def display_platform_dominance_chart(df: pd.DataFrame, brand_name: str):
    """Display radar chart showing platform dominance using unique mentions"""
    lang = st.session_state.get('language', 'en')

    # Get unique mention columns
    platforms = []
    for col in df.columns:
        if '(Unique)' in col:
            platforms.append(col.replace(' (Unique)', ''))

    # Create radar chart
    fig = go.Figure()

    # Add trace for main brand
    brand_col = get_text('brand', lang)
    main_brand_data = df[df[brand_col] == brand_name]
    if not main_brand_data.empty:
        values = [main_brand_data[f'{platform} (Unique)'].values[0] for platform in platforms]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=platforms,
            fill='toself',
            name=brand_name,
            line_color='#FF6B6B',
            fillcolor='rgba(255, 107, 107, 0.3)',
            text=[f"{v} {get_text('prompts', lang)}" for v in values],
            hovertemplate='%{theta}<br>%{text}<extra></extra>'
        ))

    # Add traces for top 3 competitors
    competitors_data = df[df[brand_col] != brand_name].head(3)
    colors = ['#4ECDC4', '#95E1D3', '#F38181']

    for idx, (_, row) in enumerate(competitors_data.iterrows()):
        values = [row[f'{platform} (Unique)'] for platform in platforms]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=platforms,
            fill='toself',
            name=row[brand_col],
            line_color=colors[idx % len(colors)],
            fillcolor=f'rgba({int(colors[idx % len(colors)][1:3], 16)}, {int(colors[idx % len(colors)][3:5], 16)}, {int(colors[idx % len(colors)][5:7], 16)}, 0.2)',
            text=[f"{v} {get_text('prompts', lang)}" for v in values],
            hovertemplate='%{theta}<br>%{text}<extra></extra>'
        ))

    # Find max value for scale - with minimum scale for readability
    max_val = 0
    for platform in platforms:
        col_name = f'{platform} (Unique)'
        max_val = max(max_val, df[col_name].max())

    # Ensure minimum scale for readability
    max_val = max(max_val, 5)  # Minimum scale of 5

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_val * 1.2],
                tickmode='linear',
                tick0=0,
                dtick=1 if max_val <= 10 else max(1, max_val // 5)
            )),
        showlegend=True,
        title={
            'text': get_text('platform_dominance_title', lang),
            'font': {'size': 16}
        },
        height=500,
        annotations=[
            dict(
                text=get_text('dominance_subtitle', lang),
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                xanchor='center',
                font=dict(size=12, color="gray")
            )
        ]
    )

    return fig


def display_competitor_heatmap(results: Dict[str, Any], brand_name: str, competitors: List[str]):
    """Display heatmap showing presence (0 or 1) across prompts and brands"""
    lang = st.session_state.get('language', 'en')

    # Prepare data for heatmap
    all_brands = [brand_name] + competitors
    platforms = list(results.keys())

    # Get total number of prompts
    total_prompts = len(results[platforms[0]]['responses'])

    # Determine display strategy based on number of prompts
    if total_prompts <= 20:
        # Show all prompts with full text
        prompts = []
        for idx, r in enumerate(results[platforms[0]]['responses']):
            prompt_text = r['prompt'][:50] if len(r['prompt']) <= 50 else r['prompt'][:47] + '...'
            prompts.append(f"{idx + 1}. {prompt_text}")
    elif total_prompts <= 50:
        # Show all prompts but with shorter text
        prompts = []
        for idx, r in enumerate(results[platforms[0]]['responses']):
            prompt_text = r['prompt'][:25] if len(r['prompt']) <= 25 else r['prompt'][:22] + '...'
            prompts.append(f"{idx + 1}. {prompt_text}")
    else:
        # For many prompts, just use numbers
        prompts = [f"{get_text('prompt', lang)} {idx + 1}" for idx in range(total_prompts)]

    # Add summary info
    st.caption(get_text('analyzing_prompts', lang, count=total_prompts, brands=len(all_brands)))

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs([
        get_text('heatmap_view', lang),
        get_text('summary_statistics', lang),
        get_text('prompt_details', lang)
    ])

    with tab1:
        # Create data matrix for each platform
        for platform in platforms:
            data_matrix = []

            for prompt_idx, response_data in enumerate(results[platform]['responses']):
                row = []

                # Brand presence (1 if mentioned, 0 if not)
                row.append(1 if response_data['mentions'] > 0 else 0)

                # Competitor presence
                for competitor in competitors:
                    comp_mentions = response_data.get('competitor_mentions', {}).get(competitor, 0)
                    row.append(1 if comp_mentions > 0 else 0)

                data_matrix.append(row)

            # Adjust figure height based on number of prompts
            fig_height = min(400 + (total_prompts * 15), 1200)  # Max height of 1200px

            # Create heatmap
            fig = px.imshow(
                data_matrix,
                labels=dict(x=get_text('brand', lang) + "s", y=get_text('prompts', lang),
                            color=get_text('present', lang)),
                x=all_brands,
                y=prompts,
                title=get_text('heatmap_title', lang, platform=platform.title()),
                color_continuous_scale=["white", "#1E88E5"],
                aspect="auto"
            )

            # Update layout to move x-axis labels to top
            fig.update_layout(
                height=fig_height,
                xaxis=dict(side="top")
            )

            # Add warning for large datasets
            if total_prompts > 50:
                st.info(f"💡 {get_text('heatmap_tip', lang, count=total_prompts)}")

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Summary statistics for better readability with many prompts
        st.subheader(f"📊 {get_text('presence_summary', lang)}")

        for platform in platforms:
            st.markdown(f"### {platform.title()}")

            # Calculate presence statistics
            summary_data = []

            for brand in all_brands:
                presence_count = 0

                if brand == brand_name:
                    presence_count = sum(1 for r in results[platform]['responses'] if r['mentions'] > 0)
                else:
                    presence_count = sum(1 for r in results[platform]['responses']
                                         if r.get('competitor_mentions', {}).get(brand, 0) > 0)

                presence_rate = (presence_count / total_prompts) * 100
                summary_data.append({
                    get_text('brand', lang): brand,
                    get_text('prompts_with_presence', lang): presence_count,
                    get_text('presence_rate', lang): f"{presence_rate:.1f}%"
                })

            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

    with tab3:
        # Detailed prompt analysis with search/filter
        st.subheader(f"🔍 {get_text('prompt_details', lang)}")

        # Add search box
        search_term = st.text_input(
            get_text('search_prompts', lang),
            placeholder=get_text('search_placeholder', lang),
            key="prompt_search"
        )

        # Platform selector with unique key to prevent reloading
        selected_platform = st.selectbox(
            get_text('select_platform', lang),
            platforms,
            format_func=lambda x: x.title(),
            key="platform_selector_heatmap"
        )

        # Filter prompts
        filtered_prompts = []
        for idx, response in enumerate(results[selected_platform]['responses']):
            if not search_term or search_term.lower() in response['prompt'].lower():
                filtered_prompts.append((idx, response))

        st.write(get_text('showing_prompts', lang, filtered=len(filtered_prompts), total=total_prompts))

        # Display filtered prompts with their brand presence
        for idx, response in filtered_prompts[:50]:  # Limit to 50 to avoid performance issues
            with st.expander(f"{get_text('prompt', lang)} {idx + 1}: {response['prompt'][:80]}..."):
                cols = st.columns(len(all_brands))

                for i, brand in enumerate(all_brands):
                    with cols[i]:
                        if brand == brand_name:
                            present = response['mentions'] > 0
                        else:
                            present = response.get('competitor_mentions', {}).get(brand, 0) > 0

                        if present:
                            st.success(f"✓ {brand}")
                        else:
                            st.text(f"✗ {brand}")

        if len(filtered_prompts) > 50:
            st.warning(get_text('showing_first_50', lang))


def display_competitor_insights(df: pd.DataFrame, brand_name: str):
    """Display key insights and recommendations based on unique mentions"""
    lang = st.session_state.get('language', 'en')

    st.subheader(f"🎯 {get_text('key_insights', lang)}")

    # Get brand rank based on unique mentions
    brand_col = get_text('brand', lang)
    brand_rank = df[df[brand_col] == brand_name].index[0] + 1
    total_brands = len(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_text = None if brand_rank == 1 else get_text('behind_leader', lang, n=brand_rank - 1)
        st.metric(
            get_text('brand_ranking', lang),
            f"#{brand_rank} of {total_brands}",
            delta=delta_text
        )

    with col2:
        # Calculate gap to leader (based on unique mentions)
        unique_col = get_text('unique_mentions', lang)
        if brand_rank > 1:
            leader_unique = df.iloc[0][unique_col]
            brand_unique = df[df[brand_col] == brand_name][unique_col].values[0]
            gap_percentage = ((leader_unique - brand_unique) / leader_unique) * 100 if leader_unique > 0 else 0
            st.metric(
                get_text('gap_to_leader', lang),
                f"{gap_percentage:.1f}%",
                delta=f"{leader_unique - brand_unique} {get_text('prompts', lang)}"
            )
        else:
            # Brand is leader - show gap to second
            if len(df) > 1:
                second_unique = df.iloc[1][unique_col]
                brand_unique = df[df[brand_col] == brand_name][unique_col].values[0]
                lead_percentage = ((brand_unique - second_unique) / brand_unique) * 100 if brand_unique > 0 else 0
                st.metric(
                    get_text('lead_over_2', lang),
                    f"{lead_percentage:.1f}%",
                    delta=f"+{brand_unique - second_unique} {get_text('prompts', lang)}"
                )

    with col3:
        # Strongest platform (based on unique mentions)
        platforms = []
        for col in df.columns:
            if '(Unique)' in col:
                platforms.append(col)

        brand_data = df[df[brand_col] == brand_name]
        if not brand_data.empty and platforms:
            platform_values = {p: brand_data[p].values[0] for p in platforms}
            strongest_platform = max(platform_values, key=platform_values.get)
            strongest_name = strongest_platform.replace(' (Unique)', '')
            st.metric(
                get_text('strongest_platform', lang),
                strongest_name,
                f"{platform_values[strongest_platform]} {get_text('prompts', lang)}"
            )

    with col4:
        # Mention density (total mentions / unique mentions)
        brand_data = df[df[brand_col] == brand_name]
        if not brand_data.empty:
            unique = brand_data[get_text('unique_mentions', lang)].values[0]
            total = brand_data[get_text('total_mentions', lang)].values[0]
            density = total / unique if unique > 0 else 0
            st.metric(
                get_text('mention_density', lang),
                f"{density:.1f}x",
                help=get_text('mention_density', lang)
            )

    # Recommendations
    st.subheader(f"💡 {get_text('recommendations', lang)}")

    vis_rate_col = get_text('visibility_rate', lang) + ' %'
    brand_visibility = df[df[brand_col] == brand_name][vis_rate_col].values[0]

    if brand_rank > 1:
        unique_col = get_text('unique_mentions', lang)
        st.info(get_text('improve_visibility', lang,
                         unique=df[df[brand_col] == brand_name][unique_col].values[0],
                         rate=brand_visibility,
                         leader_unique=df.iloc[0][unique_col],
                         leader_rate=df.iloc[0][vis_rate_col]))
    else:
        st.success(get_text('maintain_leadership', lang,
                            unique=df[df[brand_col] == brand_name][get_text('unique_mentions', lang)].values[0],
                            rate=brand_visibility))


def display_competitor_analysis_section(results: Dict[str, Any], brand_name: str, competitors: List[str]):
    """Main function to display all competitor analysis visualizations"""
    lang = st.session_state.get('language', 'en')

    if not competitors:
        st.info(get_text('no_competitors', lang))
        return

    st.header(f"🏆 {get_text('competitive_analysis', lang)}")

    # Competitor selection interface
    with st.expander(f"⚙️ {get_text('configure_competitors', lang)}", expanded=False):
        st.write(get_text('select_competitors', lang))

        # Create checkboxes for each competitor
        selected_competitors = []
        cols = st.columns(3)  # Display in 3 columns for better layout

        for idx, competitor in enumerate(competitors):
            col_idx = idx % 3
            with cols[col_idx]:
                if st.checkbox(competitor, value=True, key=f"comp_select_{competitor}"):
                    selected_competitors.append(competitor)

        # Show selected count
        st.caption(get_text('selected_count', lang, selected=len(selected_competitors), total=len(competitors)))

        # Option to add more competitors manually
        new_competitor = st.text_input(get_text('add_competitor', lang), key="add_competitor")
        if st.button(f"➕ {get_text('add', lang)}", key="add_comp_btn") and new_competitor:
            if new_competitor not in competitors and new_competitor != brand_name:
                selected_competitors.append(new_competitor)
                st.success(get_text('added_competitor', lang, competitor=new_competitor))

    # Use selected competitors for analysis
    if not selected_competitors:
        st.warning(get_text('select_one_competitor', lang))
        return

    # Create comparison table with selected competitors only
    df_comparison = create_competitor_comparison_table(results, brand_name, selected_competitors)

    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)

    brand_col = get_text('brand', lang)
    with col1:
        brand_unique = df_comparison[df_comparison[brand_col] == brand_name][get_text('unique_mentions', lang)].values[
            0]
        brand_total = df_comparison[df_comparison[brand_col] == brand_name][get_text('total_mentions', lang)].values[0]
        st.metric(get_text('your_brand_presence', lang), f"{brand_unique} {get_text('prompts', lang)}",
                  delta=f"{brand_total} {get_text('total_mentions', lang)}")

    with col2:
        avg_competitor_unique = df_comparison[df_comparison[brand_col] != brand_name][
            get_text('unique_mentions', lang)].mean()
        st.metric(get_text('avg_competitor_presence', lang), f"{avg_competitor_unique:.1f} {get_text('prompts', lang)}")

    with col3:
        brand_rank = df_comparison[df_comparison[brand_col] == brand_name].index[0] + 1
        st.metric(get_text('your_ranking', lang), f"#{brand_rank}")

    with col4:
        total_brands = len(df_comparison)
        st.metric(get_text('total_brands', lang), total_brands)

    # Display comparison table
    st.subheader(f"📊 {get_text('detailed_comparison', lang)}")

    # Create a simplified view for display
    display_columns = [brand_col, get_text('unique_mentions', lang), get_text('total_mentions', lang),
                       get_text('visibility_rate', lang) + ' %']

    # Add platform columns in a more readable format
    platforms = []
    for col in df_comparison.columns:
        if '(Unique)' in col:
            platform = col.replace(' (Unique)', '')
            platforms.append(platform)
            display_columns.insert(-3, f'{platform}')

    # Create display dataframe
    df_display = df_comparison[[brand_col, get_text('unique_mentions', lang), get_text('total_mentions', lang),
                                get_text('visibility_rate', lang) + ' %']].copy()

    for platform in platforms:
        df_display[platform] = df_comparison.apply(
            lambda row: f"{row[f'{platform} (Unique)']}/{row[f'{platform} (Total)']}",
            axis=1
        )

    # Reorder columns
    final_columns = [brand_col] + platforms + [get_text('unique_mentions', lang),
                                               get_text('total_mentions', lang),
                                               get_text('visibility_rate', lang) + ' %']
    df_display = df_display[final_columns]

    # Style the dataframe
    styled_df = df_display.style.format({
        get_text('visibility_rate', lang) + ' %': '{:.1f}%'
    })

    st.dataframe(styled_df, use_container_width=True)
    st.caption(get_text('format_note', lang))

    # Display visualizations in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"📊 {get_text('presence_comparison', lang)}",
        f"📈 {get_text('visibility_rates', lang)}",
        f"🎯 {get_text('platform_dominance', lang)}",
        f"🔥 {get_text('presence_heatmap', lang)}",
        f"📉 {get_text('total_mentions_chart', lang)}"
    ])

    with tab1:
        fig1 = display_competitor_comparison_chart(df_comparison, brand_name)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = display_visibility_rate_comparison(df_comparison, brand_name)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = display_platform_dominance_chart(df_comparison, brand_name)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        display_competitor_heatmap(results, brand_name, selected_competitors)

    with tab5:
        fig5 = display_total_mentions_chart(df_comparison, brand_name)
        st.plotly_chart(fig5, use_container_width=True)
        st.caption(get_text('all_occurrences', lang))

    # Display insights
    display_competitor_insights(df_comparison, brand_name)