"""
Stay In The Loupe - Jeweler Newsletter Generator
Flask backend API for generating monthly jeweler newsletters
"""

import os
import sys
import json
import re
import base64
import secrets
from datetime import datetime
from io import BytesIO
import pytz

# Chicago timezone for timestamps
CHICAGO_TZ = pytz.timezone('America/Chicago')

from flask import Flask, request, jsonify, send_from_directory, redirect, session, url_for, Response
from flask_cors import CORS
from authlib.integrations.flask_client import OAuth
from werkzeug.middleware.proxy_fix import ProxyFix

# SendGrid for email
try:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Email, To, Content, HtmlContent
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False
    print("[WARNING] SendGrid not installed. Email functionality disabled.")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import AI clients
from backend.integrations.openai_client import OpenAIClient
from backend.integrations.claude_client import ClaudeClient
from backend.integrations.gemini_client import GeminiClient
from backend.integrations.perplexity_client import PerplexityClient

# Import config
from config.brand_guidelines import (
    JEWELRY_NEWS_SOURCES,
    CONTENT_FILTERS,
    ONTRAPORT_CONFIG,
    TEAM_MEMBERS,
    GOOGLE_DRIVE_FOLDER_ID,
    BRAND_VOICE,
    SECTION_SPECS,
    WRITING_STYLE_GUIDE,
    BRAND_CHECK_RULES
)
from config.model_config import get_model_for_task

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Fix for running behind Cloud Run's proxy - ensures correct HTTPS URLs
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Session configuration for OAuth (Cloud Run compatible)
# Use 'or' to handle empty string case when env var is set but empty
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(32)
app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS only
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Required for OAuth redirects

# OAuth configuration
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

ALLOWED_DOMAIN = 'brite.co'

def get_current_user():
    """Get current authenticated user from session"""
    return session.get('user')

# Initialize AI clients with error handling
openai_client = None
claude_client = None
gemini_client = None
perplexity_client = None

try:
    openai_client = OpenAIClient()
    print("[OK] OpenAI initialized")
except Exception as e:
    print(f"[WARNING] OpenAI not available: {e}")

try:
    gemini_client = GeminiClient()
    if gemini_client.is_available():
        print("[OK] Gemini initialized")
    else:
        print("[WARNING] Gemini not available - add GOOGLE_AI_API_KEY")
except Exception as e:
    print(f"[WARNING] Gemini not available: {e}")

try:
    claude_client = ClaudeClient()
    print("[OK] Claude initialized")
except Exception as e:
    print(f"[WARNING] Claude not available: {e}")

try:
    perplexity_client = PerplexityClient()
    print("[OK] Perplexity initialized")
except Exception as e:
    print(f"[WARNING] Perplexity not available: {e}")

# Google Cloud Storage for drafts and images
GCS_BUCKET_NAME = 'stay-in-the-loupe-drafts'
GCS_IMAGES_BUCKET = 'stay-in-the-loupe-images'  # Public bucket for newsletter images
gcs_client = None
try:
    from google.cloud import storage as gcs_storage
    gcs_client = gcs_storage.Client()
    print("[OK] GCS initialized")
except Exception as e:
    print(f"[WARNING] GCS not available: {e}")

# Safe print function for Unicode
def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


# ============================================================================
# OAUTH AUTHENTICATION ROUTES
# ============================================================================

@app.route('/auth/login')
def auth_login():
    """Redirect to Google OAuth"""
    if get_current_user():
        return redirect('/')
    redirect_uri = url_for('auth_callback', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/auth/callback')
def auth_callback():
    """Handle OAuth callback from Google"""
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')

        if not user_info:
            return 'Failed to get user info', 400

        email = user_info.get('email', '')

        # Enforce domain restriction
        if not email.endswith(f'@{ALLOWED_DOMAIN}'):
            return f'''
            <html>
            <head><title>Access Denied</title></head>
            <body style="font-family: sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #272D3F;">
                <div style="text-align: center; color: white; padding: 2rem;">
                    <h1 style="color: #FC883A;">Access Denied</h1>
                    <p>Only @{ALLOWED_DOMAIN} email addresses are allowed.</p>
                    <p style="color: #A9C1CB;">You tried to sign in with: {email}</p>
                    <a href="/auth/login" style="color: #31D7CA;">Try again with a different account</a>
                </div>
            </body>
            </html>
            ''', 403

        # Store user in session
        session['user'] = {
            'email': email,
            'name': user_info.get('name', ''),
            'picture': user_info.get('picture', '')
        }

        return redirect('/')

    except Exception as e:
        print(f"[AUTH ERROR] OAuth callback failed: {e}")
        return f'Authentication failed: {str(e)}', 500


@app.route('/auth/logout')
def auth_logout():
    """Clear session and redirect to login"""
    session.pop('user', None)
    return redirect('/auth/login')


# ============================================================================
# ROUTES - STATIC FILES
# ============================================================================

@app.route('/')
def serve_index():
    """Serve the main app with auth check"""
    user = get_current_user()
    if not user:
        return redirect('/auth/login')

    # Read and serve the index.html with user info injected
    with open('index.html', 'r', encoding='utf-8') as f:
        html = f.read()

    # Inject user info for the frontend
    user_script = f'''<script>
    window.AUTH_USER = {json.dumps(user)};
    </script>
</head>'''
    html = html.replace('</head>', user_script, 1)

    return Response(html, mimetype='text/html')

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "app": "Stay In The Loupe", "timestamp": datetime.now().isoformat()})


# ============================================================================
# HELPER FUNCTIONS - LLM ENRICHMENT (Matching BriteCo Brief)
# ============================================================================

def enrich_results_with_llm(results: list, original_query: str, section: str = 'general') -> list:
    """
    Use LLM to generate newsletter-ready content from research results.
    Produces three-section format: headline, industry_data, so_what
    Section-specific prompts for The Good, The Bad, The Ugly.
    """
    if not results:
        return results

    try:
        model_config = get_model_for_task('research_enrichment')
        model_id = model_config.get('id', 'gpt-5.2')
        max_tokens_param = model_config.get('max_tokens_param', 'max_tokens')

        safe_print(f"[Enrichment] Using model: {model_id} for section: {section}")

        results_text = ""
        for i, r in enumerate(results):
            results_text += f"""
Result {i+1}:
- URL: {r.get('url', '')}
- Publisher: {r.get('publisher', '')}
- Raw snippet: {r.get('snippet', '')[:500]}
"""

        # Section-specific prompts from style guide
        section_prompts = {
            'the_good': """You are analyzing research findings for "The Good" section of a jewelry newsletter.
This section features POSITIVE jewelry news: new designs, trends, sales records, success stories, innovations, positive industry developments.

Focus on uplifting, inspiring stories that jewelers can share with their customers or use to feel good about the industry.
Headlines should be catchy and fun with wordplay when possible (max 8 words).
Copy should be brief but impactful (max 30 words).
Always include the source link relevance.""",

            'the_bad': """You are analyzing research findings for "The Bad" section of a jewelry newsletter.
This section features CAUTIONARY TALES: heists, thefts, scams, fraud, market downturns, security breaches, negative news that jewelers should know about.

Focus on stories that serve as warnings or lessons for jewelers to protect their business.
Headlines should be catchy with a serious undertone (max 8 words).
Copy should be brief but impactful (max 30 words).
Always include the source link relevance.""",

            'the_ugly': """You are analyzing research findings for "The Ugly" section of a jewelry newsletter.
This section features BIZARRE, UNUSUAL, or EYEBROW-RAISING jewelry stories: weird finds, strange news, unusual circumstances, quirky stories, celebrity jewelry drama, odd discoveries.

Focus on stories that are entertaining, surprising, or make people say "wow, really?"
Headlines should be catchy and fun with humor when appropriate (max 8 words).
Copy should be brief but impactful (max 30 words).
Always include the source link relevance."""
        }

        section_context = section_prompts.get(section, f"""You are analyzing research findings for a jewelry industry newsletter. The user searched for: "{original_query}"

Focus on content relevant to jewelers and their business.""")

        prompt = f"""{section_context}

Here are research findings to transform into newsletter-ready content:
{results_text}

For EACH result, extract/generate:
1. headline: A compelling newsletter headline (5-8 words max, catchy with wordplay)
2. industry_data: The key fact or story hook (1-2 sentences, max 30 words). Extract the most interesting detail.
3. so_what: Why this matters to jewelers (1 actionable sentence)
4. impact: HIGH (must read), MEDIUM (interesting), or LOW (nice to know)

Return a JSON array with exactly {len(results)} objects:
[
  {{"headline": "...", "industry_data": "...", "so_what": "...", "impact": "HIGH|MEDIUM|LOW"}},
  ...
]

Guidelines:
- Headlines should be catchy, clever, with wordplay when possible
- industry_data should be the compelling story hook, not dry facts
- so_what should connect to jeweler relevance
- Prioritize stories that fit the section theme perfectly

Return ONLY the JSON array, no other text."""

        api_params = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        api_params[max_tokens_param] = 2000

        response = openai_client.client.chat.completions.create(**api_params)
        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n", "", content)
            content = re.sub(r"\n```$", "", content).strip()

        enriched = json.loads(content)

        for i, r in enumerate(results):
            if i < len(enriched):
                r['headline'] = enriched[i].get('headline', r.get('title', ''))
                r['title'] = r['headline']
                r['industry_data'] = enriched[i].get('industry_data', r.get('snippet', ''))
                r['so_what'] = enriched[i].get('so_what', '')
                r['impact'] = enriched[i].get('impact', 'MEDIUM')
                r['snippet'] = r['industry_data']

        impact_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        results.sort(key=lambda x: impact_order.get(x.get('impact', 'LOW'), 2))

        safe_print(f"[LLM Enrichment] Successfully enriched {len(results)} results")
        return results

    except Exception as e:
        safe_print(f"[LLM Enrichment] Error: {e} - returning original results")
        import traceback
        traceback.print_exc()
        return results


def analyze_industry_impact(results: list) -> list:
    """
    Use LLM to analyze each result for jewelry industry impact.
    Generates newsletter-ready headlines and impact scores.
    """
    if not results:
        return results

    try:
        model_config = get_model_for_task('research_enrichment')
        model_id = model_config.get('id', 'gpt-5.2')
        max_tokens_param = model_config.get('max_tokens_param', 'max_tokens')

        safe_print(f"[Insight Builder] Analyzing {len(results)} results with {model_id}...")

        results_text = ""
        for i, r in enumerate(results):
            results_text += f"""
Result {i+1}:
- Signal: {r.get('signal_source', 'unknown')}
- Publisher: {r.get('publisher', '')}
- Raw title: {r.get('title', '')[:100]}
- Snippet: {r.get('description', r.get('snippet', ''))[:400]}
"""

        prompt = f"""You are analyzing news articles for a jewelry industry newsletter.

For each article, determine its impact on jewelers and their business.

Here are the articles:
{results_text}

For EACH article, provide:
1. headline: A newsletter-ready headline (5-12 words, actionable for jewelers)
2. impact: HIGH (immediate action needed), MEDIUM (worth monitoring), or LOW (FYI only)
3. signals: Array of affected categories from [gold_prices, diamond_market, luxury_trends, retail, design_trends, economic, heists_security, technology]
4. so_what: One sentence explaining what jewelers should do about this

Return a JSON array with exactly {len(results)} objects:
[
  {{"headline": "...", "impact": "HIGH|MEDIUM|LOW", "signals": ["..."], "so_what": "..."}},
  ...
]

Guidelines:
- HIGH impact: significant price changes, market shifts, security alerts
- MEDIUM impact: emerging trends, technology changes, forecasts
- LOW impact: general news, minor updates

Return ONLY the JSON array, no other text."""

        api_params = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        api_params[max_tokens_param] = 2000

        response = openai_client.client.chat.completions.create(**api_params)
        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n", "", content)
            content = re.sub(r"\n```$", "", content).strip()

        enriched = json.loads(content)

        for i, r in enumerate(results):
            if i < len(enriched):
                r['headline'] = enriched[i].get('headline', r.get('title', ''))
                r['impact'] = enriched[i].get('impact', 'MEDIUM')
                r['signals'] = enriched[i].get('signals', [])
                r['so_what'] = enriched[i].get('so_what', '')
                r['industry_data'] = r.get('description', r.get('snippet', ''))

        impact_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        results.sort(key=lambda x: impact_order.get(x.get('impact', 'LOW'), 2))

        safe_print(f"[Insight Builder] Analysis complete - enriched {len(results)} results")
        return results

    except Exception as e:
        safe_print(f"[Insight Builder] Analysis error: {e} - returning original results")
        for r in results:
            r['headline'] = r.get('title', 'Industry Update')
            r['impact'] = 'MEDIUM'
            r['signals'] = [r.get('signal_source', 'general')]
            r['so_what'] = 'Monitor this trend for potential business impact.'
        return results


def analyze_story_angles(results: list, user_query: str) -> list:
    """
    Use LLM to analyze articles and surface interesting story angles for newsletters.
    """
    if not results:
        return results

    try:
        model_config = get_model_for_task('research_enrichment')
        model_id = model_config.get('id', 'gpt-5.2')
        max_tokens_param = model_config.get('max_tokens_param', 'max_tokens')

        safe_print(f"[Source Explorer] Analyzing {len(results)} results with {model_id}...")

        results_text = ""
        for i, r in enumerate(results):
            results_text += f"""
Article {i+1}:
- Title: {r.get('title', '')[:100]}
- Publisher: {r.get('publisher', '')}
- Snippet: {r.get('snippet', r.get('description', ''))[:400]}
"""

        prompt = f"""You are a newsletter editor for jewelers. The user searched for: "{user_query}"

Analyze these articles and surface the most interesting story angles for a jeweler newsletter.

Here are the articles:
{results_text}

For EACH article, provide:
1. story_angle: A compelling newsletter story angle (1-2 sentences) - what's the interesting hook for jewelers?
2. headline: A catchy headline (5-10 words) that would grab a jeweler's attention
3. why_it_matters: One sentence on why jewelers should care about this
4. content_type: One of [trend, tip, news, insight, case_study]
5. impact: HIGH, MEDIUM, or LOW

Return a JSON array with exactly {len(results)} objects:
[
  {{"story_angle": "...", "headline": "...", "why_it_matters": "...", "content_type": "...", "impact": "MEDIUM"}},
  ...
]

Guidelines:
- Focus on actionable insights jewelers can use with customers
- Look for data points, trends, or tips that can be turned into content
- Headlines should be specific and engaging (not generic)

Return ONLY the JSON array, no other text."""

        api_params = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        }
        api_params[max_tokens_param] = 2000

        response = openai_client.client.chat.completions.create(**api_params)
        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n", "", content)
            content = re.sub(r"\n```$", "", content).strip()

        enriched = json.loads(content)

        for i, r in enumerate(results):
            if i < len(enriched):
                r['story_angle'] = enriched[i].get('story_angle', '')
                r['headline'] = enriched[i].get('headline', r.get('title', ''))
                r['why_it_matters'] = enriched[i].get('why_it_matters', '')
                r['content_type'] = enriched[i].get('content_type', 'insight')
                r['so_what'] = enriched[i].get('why_it_matters', r.get('so_what', ''))
                r['industry_data'] = r.get('snippet', r.get('description', ''))
                r['impact'] = enriched[i].get('impact', 'MEDIUM')

        safe_print(f"[Source Explorer] Story analysis complete - enriched {len(results)} results")
        return results

    except Exception as e:
        safe_print(f"[Source Explorer] Analysis error: {e} - returning original results")
        for r in results:
            r['story_angle'] = r.get('snippet', '')[:150]
            r['headline'] = r.get('title', 'Industry Update')
            r['why_it_matters'] = 'Review this article for potential newsletter content.'
            r['content_type'] = 'insight'
            r['impact'] = 'MEDIUM'
        return results


def transform_to_shared_schema(results: list, source_card: str) -> list:
    """Transform results to shared schema matching BriteCo Brief"""
    return [{
        'title': r.get('title', ''),
        'headline': r.get('headline', r.get('title', '')),
        'url': r.get('url', r.get('source_url', '')),
        'publisher': r.get('publisher', ''),
        'published_at': r.get('published_date', r.get('published_at', '')),
        'snippet': r.get('snippet', r.get('description', '')),
        'industry_data': r.get('industry_data', r.get('snippet', r.get('description', ''))),
        'so_what': r.get('so_what', ''),
        'source_card': source_card,
        'content_type': r.get('content_type', 'news'),
        'impact': r.get('impact', 'MEDIUM'),
        'signals': r.get('signals', []),
        'signal_source': r.get('signal_source', '')
    } for r in results]


# ============================================================================
# ROUTES - RESEARCH (Perplexity, Insights, Source Explorer)
# ============================================================================

@app.route('/api/v2/search-perplexity', methods=['POST'])
def search_perplexity_v2():
    """
    Perplexity Research Card - uses Perplexity sonar model for research with citations
    Section-specific searches for The Good, The Bad, The Ugly
    """
    try:
        data = request.json
        query = data.get('query', 'jewelry industry news')
        time_window = data.get('time_window', '30d')
        section = data.get('section', 'general')
        exclude_urls = data.get('exclude_urls', [])

        safe_print(f"\n[API v2] Perplexity Research: query='{query}', section={section}, time_window={time_window}")

        # Check if Perplexity is available
        if not perplexity_client or not perplexity_client.is_available():
            return jsonify({
                'success': False,
                'error': 'Perplexity API not configured. Add PERPLEXITY_API_KEY to .env',
                'results': []
            }), 503

        # Section-specific search queries based on style guide
        section_queries = {
            'the_good': 'positive jewelry news new designs trends sales records success stories innovations awards',
            'the_bad': 'jewelry heists thefts scams fraud robbery security breaches crime negative news',
            'the_ugly': 'bizarre unusual strange jewelry design pieces weird jewelry ugly jewelry specific jewelry item quirky celebrity jewelry strange jewelry finds',
            'partner_advantage': 'jewelry business advice tips best practices how to jeweler marketing sales strategies recommendations for jewelers retail tips',
            'industry_pulse': 'jewelry industry news trends market analysis business developments',
            'industry_news': 'jewelry industry latest news headlines announcements events'
        }

        # Build section-specific query
        section_focus = section_queries.get(section, '')
        if section_focus:
            # For Partner Advantage, use more specific advice-focused query
            if section == 'partner_advantage':
                search_query = f"practical tips and advice for jewelry retail {query} jeweler best practices how to guides"
            else:
                search_query = f"jewelry {section_focus} {query}"
        else:
            search_query = f"jewelry industry {query}"

        safe_print(f"[API v2] Search query: {search_query}")

        # Search using Perplexity
        search_results = perplexity_client.search(
            query=search_query,
            time_window=time_window,
            max_results=8
        )

        # Filter out excluded URLs
        if exclude_urls:
            search_results = [r for r in search_results if r.get('url') not in exclude_urls]

        # Take top 8 results
        results = search_results[:8]

        # Enrich results with LLM-generated titles and jeweler guidance (section-specific)
        if results:
            safe_print(f"[API v2] Enriching {len(results)} Perplexity results for section: {section}")
            results = enrich_results_with_llm(results, query, section)

        # Build query description for UI
        time_desc = {
            '7d': 'past week',
            '15d': 'past 2 weeks',
            '30d': 'past month',
            '90d': 'past 3 months'
        }.get(time_window, 'recent')

        section_label = {
            'the_good': 'The Good (positive news)',
            'the_bad': 'The Bad (cautionary tales)',
            'the_ugly': 'The Ugly (bizarre stories)'
        }.get(section, 'general news')

        return jsonify({
            'success': True,
            'results': results,
            'queries_used': [f"Jewelry {section_label} from {time_desc}: {query}"],
            'source': 'perplexity',
            'section': section,
            'generated_at': datetime.now().isoformat()
        })

    except Exception as e:
        safe_print(f"[API v2 ERROR] Perplexity Research: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e), 'results': []}), 500


@app.route('/api/v2/search-insights', methods=['POST'])
def search_insights_v2():
    """
    Insight Builder Card - searches ALL 8 signals and analyzes industry impact
    """
    try:
        data = request.json
        time_window = data.get('time_window', '30d')
        exclude_urls = data.get('exclude_urls', [])

        safe_print(f"\n[API v2] Insight Builder: Searching ALL 8 signals")

        # Jewelry-specific signal queries
        SIGNAL_QUERIES = {
            'gold_prices': 'US gold prices market precious metals jewelry industry recent news',
            'diamond_market': 'US diamond prices market trends lab-grown natural recent',
            'luxury_trends': 'US luxury jewelry market consumer spending trends recent',
            'retail': 'US jewelry retail sales trends brick mortar online recent',
            'design_trends': 'jewelry design trends fashion styles 2024 2025 recent',
            'economic': 'US consumer spending economy jewelry luxury goods recent',
            'heists_security': 'jewelry heists theft security robbery crime recent news',
            'technology': 'jewelry technology 3D printing CAD design innovation recent'
        }

        all_results = []
        seen_urls = set(exclude_urls)

        safe_print(f"[Insight Builder] Searching all 8 jewelry signals...")

        for signal, query_terms in SIGNAL_QUERIES.items():
            try:
                prompt = f"""Search for recent US news about {signal.replace('_', ' ')} in jewelry industry.

Find articles about the United States with data points, statistics, and business impact.
Focus on jewelry retail and wholesale markets.
Search terms: {query_terms}

Return results with title, url, publisher, published_date, and summary with key data points."""

                results = openai_client.search_web_responses_api(prompt, max_results=4, exclude_urls=list(seen_urls))

                for r in results:
                    url = r.get('url', '')
                    if url and url not in seen_urls:
                        r['signal_source'] = signal
                        all_results.append(r)
                        seen_urls.add(url)

                safe_print(f"[Insight Builder] Signal '{signal}' returned {len(results)} results")

            except Exception as e:
                safe_print(f"[Insight Builder] Error searching signal '{signal}': {e}")
                continue

        safe_print(f"[Insight Builder] Total unique results: {len(all_results)}")

        # Analyze results with GPT for industry impact
        enriched_results = analyze_industry_impact(all_results)

        # Transform to shared schema
        results = transform_to_shared_schema(enriched_results, 'insight')

        # Get signals searched
        signals_searched = list(SIGNAL_QUERIES.keys())

        return jsonify({
            'success': True,
            'results': results[:12],
            'signals_searched': signals_searched,
            'source': 'insight',
            'generated_at': datetime.now().isoformat()
        })

    except Exception as e:
        safe_print(f"[API v2 ERROR] Insight Builder: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e), 'results': []}), 500


@app.route('/api/v2/search-sources', methods=['POST'])
def search_sources_v2():
    """
    Source Explorer Card - searches specific industry sites with 3-query cascade
    """
    try:
        data = request.json
        query = data.get('query', 'jewelry industry news')
        source_packs = data.get('source_packs', ['jewelry'])
        time_window = data.get('time_window', '30d')
        section = data.get('section', 'general')
        exclude_urls = data.get('exclude_urls', [])

        safe_print(f"\n[API v2] Source Explorer: query='{query}', section={section}, packs={source_packs}, time_window={time_window}")

        # Convert time window to human-readable for query
        time_desc = {
            '7d': 'past week',
            '15d': 'past 2 weeks',
            '30d': 'past month',
            '90d': 'past 3 months'
        }.get(time_window, 'recent')

        # Jewelry industry source packs
        SITE_PACKS = {
            'jewelry': JEWELRY_NEWS_SOURCES,
            'luxury': [
                'luxurydaily.com', 'jckonline.com', 'nationaljeweler.com',
                'rapaport.com', 'professionaljeweler.com'
            ],
            'retail': [
                'retaildive.com', 'chainstoreage.com', 'jckonline.com',
                'nationaljeweler.com'
            ],
            'design': [
                'jckonline.com', 'nationaljeweler.com', 'jewellerynet.com',
                'rapaport.com'
            ]
        }

        # Collect sites from selected packs
        sites = []
        for pack in source_packs:
            sites.extend(SITE_PACKS.get(pack, []))
        sites = list(set(sites))

        if not sites:
            sites = JEWELRY_NEWS_SOURCES

        # Build site: queries with 3-query cascade
        site_query = ' OR '.join([f'site:{s}' for s in sites[:6]])

        # Section-specific queries
        if section == 'partner_advantage':
            # For Partner Advantage, search for advice/tips articles
            queries = [
                f"""Search for: ({site_query}) {query} tips advice how to best practices

Find practical advice articles from the {time_desc} from jewelry industry sources.
Focus on actionable tips, strategies, and best practices for jewelers.
Return results with title, url, publisher, published_date, and summary.""",

                f"""Search for: ({site_query}) jeweler business advice marketing sales tips strategies

Find how-to guides and advice articles from the {time_desc} about running a jewelry business.
Return results with title, url, publisher, published_date, and summary.""",

                f"""Search for jewelry retail tips best practices from industry experts.

Find advice articles from the {time_desc} about: {query}
Focus on practical recommendations and strategies for jewelers.
Return results with title, url, publisher, published_date, and summary."""
            ]
        else:
            # Default queries for news sections
            queries = [
                f"""Search for: ({site_query}) {query}

Find articles from the {time_desc} from these jewelry industry sources.
Return results with title, url, publisher, published_date, and summary.""",

                f"""Search for: ({site_query}) jewelry industry news trends

Find business news from the {time_desc} about jewelry retail and wholesale.
Return results with title, url, publisher, published_date, and summary.""",

                f"""Search for jewelry industry news from trade publications.

Find articles from the {time_desc} about: {query}
Focus on business insights, trends, and industry analysis.
Return results with title, url, publisher, published_date, and summary."""
            ]

        safe_print(f"[API v2] Source Explorer using {len(sites)} sites from packs: {source_packs}")

        # Multi-search with cascade
        all_results = []
        seen_urls = set(exclude_urls)

        for q in queries:
            if len(all_results) >= 8:
                break

            try:
                results = openai_client.search_web_responses_api(
                    q,
                    max_results=6,
                    exclude_urls=list(seen_urls)
                )

                for r in results:
                    url = r.get('url', '')
                    if url and url not in seen_urls:
                        all_results.append(r)
                        seen_urls.add(url)

            except Exception as e:
                safe_print(f"[API v2] Source query error: {e}")
                continue

        # Transform to shared schema
        results = transform_to_shared_schema(all_results[:8], 'explorer')

        # Enrich with GPT story angle analysis
        results = analyze_story_angles(results, query)

        # Query summaries for UI display
        query_summaries = [
            f"1. Site-specific: {query} from {', '.join(sites[:3])}...",
            "2. Broader: jewelry industry news from sites",
            "3. Fallback: jewelry news (any source)"
        ]

        return jsonify({
            'success': True,
            'results': results,
            'queries_used': query_summaries,
            'source_packs': source_packs,
            'source': 'explorer',
            'generated_at': datetime.now().isoformat()
        })

    except Exception as e:
        safe_print(f"[API v2 ERROR] Source Explorer: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e), 'results': []}), 500


def extract_domain(url):
    """Extract domain from URL for publisher name"""
    if not url:
        return ''
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        return domain
    except:
        return ''




# ============================================================================
# ROUTES - CONTENT GENERATION
# ============================================================================

@app.route('/api/research-articles', methods=['POST'])
def research_articles():
    """Deep research on selected articles using GPT"""
    try:
        data = request.json
        articles = data.get('articles', {})
        month = data.get('month', '')

        safe_print(f"\n[API] Researching articles for {month}...")

        researched = {}

        for section, article_data in articles.items():
            if not article_data:
                continue

            # Handle lists of articles (industry_pulse, partner_advantage)
            if isinstance(article_data, list):
                if not article_data:
                    continue
                safe_print(f"  Researching {section}: {len(article_data)} articles...")

                researched_list = []
                for i, art in enumerate(article_data[:5]):  # Limit to first 5
                    if not art or not isinstance(art, dict):
                        continue
                    title = art.get('title', '')
                    url = art.get('url', '')
                    snippet = art.get('snippet', '')

                    prompt = f"""Research this article for a jewelry newsletter:

Title: {title}
URL: {url}
Snippet: {snippet}

Provide a concise summary (75-100 words) covering the key facts and why it matters to jewelry professionals."""

                    try:
                        response = openai_client.client.chat.completions.create(
                            model="gpt-5.2",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.5,
                            max_completion_tokens=200
                        )
                        researched_list.append({
                            'title': title,
                            'url': url,
                            'research': response.choices[0].message.content.strip()
                        })
                    except Exception as e:
                        safe_print(f"    Error researching article {i+1}: {e}")
                        researched_list.append({
                            'title': title,
                            'url': url,
                            'research': snippet
                        })

                researched[section] = researched_list
                continue

            # Handle single article (dict) - the_good, the_bad, the_ugly
            article = article_data
            if not isinstance(article, dict):
                continue

            safe_print(f"  Researching {section}: {article.get('title', '')[:50]}...")

            prompt = f"""Research this article for a jewelry newsletter:

Title: {article.get('title', '')}
URL: {article.get('url', '')}
Snippet: {article.get('snippet', '')}

Provide a detailed summary (200-300 words) covering:
1. Key facts and statistics
2. Why this matters to jewelry professionals
3. Implications for the industry

Write in a professional but engaging tone."""

            try:
                response = openai_client.client.chat.completions.create(
                    model="gpt-5.2",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_completion_tokens=500
                )

                researched[section] = {
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'research': response.choices[0].message.content.strip()
                }

            except Exception as e:
                safe_print(f"  Error researching {section}: {e}")
                researched[section] = {
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'research': article.get('snippet', '')
                }

        return jsonify({
            'success': True,
            'researched': researched
        })

    except Exception as e:
        safe_print(f"[API ERROR] Research: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rewrite-content', methods=['POST'])
def rewrite_content():
    """AI rewrite for intro and brite spot content with strict word limits"""
    try:
        data = request.json
        content = data.get('content', '')
        tone = data.get('tone', 'professional')
        section = data.get('section', 'intro')

        safe_print(f"\n[API] AI Rewrite: section={section}, tone={tone}")

        tone_prompts = {
            'professional': 'Use a professional, authoritative tone that conveys expertise.',
            'friendly': 'Use a warm, approachable tone that feels personable and engaging.',
            'exciting': 'Use an enthusiastic, energetic tone that creates excitement.',
            'informative': 'Use a clear, educational tone that emphasizes key facts.',
            'witty': 'Use a clever, witty tone with subtle humor — smart wordplay welcome but keep it tasteful and professional. Think confident newsletter voice, not comedy routine.'
        }

        # Section-specific context and STRICT word limits
        section_specs = {
            'intro': {
                'context': 'This is the newsletter introduction that welcomes readers to the monthly edition.',
                'max_words': 50,
                'description': '2-3 sentences maximum'
            },
            'brite_spot': {
                'context': 'This is "The Brite Spot" section highlighting BriteCo/company news and announcements.',
                'max_words': 100,
                'description': '3-4 sentences maximum'
            },
            'brite_spot_title': {
                'context': 'This is the sub-header title for "The Brite Spot" section.',
                'max_words': 15,
                'description': '1 short headline (under 15 words)'
            }
        }

        spec = section_specs.get(section, section_specs['intro'])

        rewrite_system = """You are a professional newsletter copywriter for "Stay In The Loupe," a monthly jewelry industry newsletter by BriteCo.

=== REAL EXAMPLES OF NEWSLETTER VOICE (match this conversational style) ===

INTRO EXAMPLES:
- "You asked, and BriteCo delivers: Now you can choose from four different appraisal designs to create the look that suits you best. Plus, keep reading to discover findings from our lab-grown vs. natural diamond report, get the latest on tariffs affecting the industry, and find tips on promoting self-purchases."
- "The holidays are coming, and we're unwrapping the biggest trends to watch out for this season -- including how to gift yourself with prominent sales opportunities. Plus, we take a closer look at the benefits of Google Ads vs. social media ads and when to utilize each one."
- "Did you know BriteCo has a thriving blog and YouTube channel? As we continue to ramp up content, we are looking for expert voices that can share advice on a range of jewelry topics."

BRITE SPOT EXAMPLES:
- "We are wishing you a very happy holiday season and thank you for another great year of partnership. We value your contributions towards our mission of helping clients protect their jewelry."
- "We want to prioritize the next wave of POS integrations and need your help to do so. Please fill out our quick and easy form and tell us your preferred point-of-sale system. When you do, you'll be entered to win a $50 gift card!"
- "BriteCo is building up a database of expert voices to include in our blog and YouTube channel. Every time you're interviewed, we'll give you $20 and your name and website will be featured."

=== END EXAMPLES ===

VOICE:
- Conversational and warm -- like writing to a colleague
- Use active voice and contractions naturally (we're, you'll, don't)
- Include specific details (names, numbers, percentages) when available
- AVOID: "leverage", "robust", "comprehensive", "in today's evolving", generic corporate-speak"""

        rewrite_prompt = f"""Rewrite this content for the newsletter.

{spec['context']}

TONE: {tone_prompts.get(tone, tone_prompts['professional'])}

CRITICAL WORD LIMIT: Maximum {spec['max_words']} words ({spec['description']})

Original content:
{content}

Keep the core message. STRICTLY stay within {spec['max_words']} words. Output ONLY the rewritten content."""

        try:
            response = claude_client.generate_content(
                prompt=rewrite_prompt,
                system_prompt=rewrite_system,
                max_tokens=500,
                temperature=0.7
            )
            rewritten = response.get('content', '').strip()
        except Exception as e:
            safe_print(f"[API] Claude error, falling back to OpenAI: {e}")
            response = openai_client.client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=500
            )
            rewritten = response.choices[0].message.content.strip()

        # Strip markdown formatting for textarea display (don't convert to HTML)
        # Remove **bold** markers but keep the text
        import re
        rewritten = re.sub(r'\*\*([^*]+)\*\*', r'\1', rewritten)
        # Remove *italic* markers but keep the text
        rewritten = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', rewritten)

        return jsonify({
            'success': True,
            'rewritten': rewritten,
            'original': content,
            'tone': tone
        })

    except Exception as e:
        safe_print(f"[API ERROR] Rewrite: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rewrite-section', methods=['POST'])
def rewrite_section():
    """Rewrite or generate newsletter section content using Claude"""
    try:
        data = request.json
        content = data.get('content', '')
        section = data.get('section', '')
        tone = data.get('tone', 'professional')

        if not content:
            return jsonify({'success': False, 'error': 'Content required'}), 400

        tone_instructions = {
            'witty': 'Use a clever, witty tone with subtle humor.',
            'friendly': 'Use a warm, conversational, and approachable tone.',
            'exciting': 'Make it energetic and exciting with action words.',
            'informative': 'Keep it clear, factual, and professional.',
            'professional': 'Use formal business language and tone.'
        }
        tone_line = f"\n- TONE: {tone_instructions.get(tone, '')}" if tone else ""

        if not claude_client:
            return jsonify({'success': False, 'error': 'Claude client not available'}), 500

        section_rewrite_system = """You are a professional newsletter copywriter for "Stay In The Loupe," a jewelry industry newsletter by BriteCo.

=== REAL EXAMPLES OF NEWSLETTER VOICE ===
- "Over the past five years, lab-grown diamonds have fundamentally reshaped the diamond jewelry industry, evolving from a niche product into a mainstream choice. Today, they account for more than 45% of all US engagement ring purchases. We recently released a report with proprietary data that reveals significant pricing, style, and consumer purchase behavior changes."
- "As 2025 winds down, we wanted to wish you and yours a very happy holiday season and thank you for another great year of partnership. We value your contributions towards our mission of helping clients protect their jewelry."
- "BriteCo is building up a database of expert voices to include in our blog and YouTube channel on topics ranging from how to care for jewelry to the latest diamond trends and more. Every time you're interviewed, we'll give you $20 and your name and website will be featured."
=== END EXAMPLES ===

VOICE:
- Sound conversational -- like writing to a colleague
- Use contractions naturally (it's, we're, you'll)
- Include specific details (names, numbers, percentages) when available
- AVOID: "In today's ever-evolving landscape", "leverage", "robust", "comprehensive"
- No markdown formatting — output plain text only"""

        section_rewrite_prompt = f"""Rewrite this section content.

{content}

GUIDELINES:
- Keep it concise: 2-3 short paragraphs, under 100 words total
- Tone: professional, warm, and engaging{tone_line}
- Focus on value to jewelers and jewelry retailers

Output ONLY the content, no labels or explanations."""

        result = claude_client.generate_content(
            prompt=section_rewrite_prompt,
            system_prompt=section_rewrite_system,
            max_tokens=400,
            temperature=0.6
        )

        rewritten = result.get('content', '').strip()

        return jsonify({
            'success': True,
            'rewritten': rewritten,
            'original': content,
            'section': section
        })

    except Exception as e:
        safe_print(f"[API ERROR] Section rewrite: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def convert_markdown_to_html(text):
    """Convert markdown formatting to HTML (bold, italic, links)"""
    import re
    if not text:
        return text
    text = str(text)
    # Convert markdown links [text](url) to HTML links
    text = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        r'<a href="\2" target="_blank" style="color: #008181; text-decoration: underline;">\1</a>',
        text
    )
    # Convert **bold** to <strong>
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    # Convert *italic* to <em> (but not if it's already part of bold)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    return text

def process_generated_content(content):
    """Process generated content to convert markdown to HTML"""
    if isinstance(content, dict):
        return {k: process_generated_content(v) for k, v in content.items()}
    elif isinstance(content, list):
        return [process_generated_content(item) for item in content]
    elif isinstance(content, str):
        return convert_markdown_to_html(content)
    return content


@app.route('/api/generate-newsletter', methods=['POST'])
def generate_newsletter():
    """Generate newsletter content using Claude Opus 4.5"""
    try:
        data = request.json
        month = data.get('month', '')
        sections_data = data.get('sections', {})
        research_data = data.get('research', {})  # GPT-5.2 research results
        brite_spot_content = data.get('brite_spot', '')  # Optional - will auto-generate if empty
        intro_content = data.get('intro', '')  # Optional - will auto-generate if empty

        safe_print(f"\n[API] Generating newsletter content for {month}...")
        safe_print(f"  Auto-generate intro: {not intro_content}")
        safe_print(f"  Auto-generate brite spot: {not brite_spot_content}")
        safe_print(f"  sections_data keys: {list(sections_data.keys())}")
        safe_print(f"  Industry Pulse in sections_data: {'industry_pulse' in sections_data}")
        safe_print(f"  Industry Pulse value type: {type(sections_data.get('industry_pulse'))}")
        safe_print(f"  Industry Pulse value: {sections_data.get('industry_pulse')}")
        safe_print(f"  Industry Pulse articles received: {len(sections_data.get('industry_pulse', []))}")
        if sections_data.get('industry_pulse'):
            safe_print(f"  Industry Pulse article titles: {[a.get('title', 'No title')[:50] for a in sections_data['industry_pulse']]}")

        # Merge research results into sections data
        for section_key, research in research_data.items():
            if section_key in sections_data and sections_data[section_key]:
                # Handle list research (for industry_pulse, partner_advantage)
                if isinstance(research, list):
                    # For list sections, merge research into each article
                    section_articles = sections_data[section_key]
                    if isinstance(section_articles, list):
                        for i, art in enumerate(section_articles):
                            if i < len(research) and isinstance(art, dict):
                                art['research'] = research[i].get('research', '') if isinstance(research[i], dict) else ''
                    safe_print(f"  Merged research for {section_key}: {len(research)} articles")
                elif isinstance(research, dict):
                    # Single article research
                    if isinstance(sections_data[section_key], dict):
                        sections_data[section_key]['research'] = research.get('research', '')
                        safe_print(f"  Merged research for {section_key}: {len(research.get('research', ''))} chars")

        generated = {
            'intro': intro_content,
            'brite_spot': brite_spot_content
        }

        # Generate The Good, The Bad, The Ugly
        for section_key in ['the_good', 'the_bad', 'the_ugly']:
            if section_key in sections_data and sections_data[section_key]:
                article = sections_data[section_key]

                section_type = {
                    'the_good': 'positive, uplifting jewelry news',
                    'the_bad': 'cautionary tale about jewelry heists, thefts, or scams',
                    'the_ugly': 'bizarre or unusual jewelry story'
                }

                gbu_system = """You write "The Good, The Bad, The Ugly" section for "Stay In The Loupe," a jewelry industry newsletter.

=== REAL EXAMPLES FROM PAST ISSUES (match this punchy, specific voice) ===

GOOD examples:
- Subtitle: "A Ring That Says 'You Belong With Me'" / Copy: "Taylor Swift's engagement news may not be shocking anymore, but her ring still is; estimates say the 18k gold diamond stunner is worth more than $1 million."
- Subtitle: "A Gold Rush" / Copy: "As the value of gold continues to reach new milestones in 2025, JPMorgan Chase CEO Jamie Dimon believes sales could 'easily' climb to $10,000 per ounce in the near future."
- Subtitle: "It's Alive!" / Copy: "A monster-sized collection from Tiffany & Co. gets the spotlight in Guillermo Del Toro's new film adaptation of Frankenstein, out in November. The movie features 27 pieces, including some from its archives."

BAD examples:
- Subtitle: "Caught in a Lab-Grown Lie" / Copy: "A New Jersey jeweler was arrested in August and charged with defrauding a customer of $23,000 after selling lab-grown diamonds as natural gems."
- Subtitle: "A Takeover Shakeup" / Copy: "Employees at Heller Jewelers in San Ramon, California had a huge scare in September as 20 masked and armed thieves broke in and fired several rounds. Thankfully, no one was injured."
- Subtitle: "The Louvre's Losses" / Copy: "Seven arrests have been made in the brazen heist that occurred at the Louvre in October. Thieves got away with eight precious jewels, worth up to $100 million, which have yet to be recovered."

UGLY examples:
- Subtitle: "A Big Call for Help" / Copy: "Well, that's one way to get attention. This 2013 statement piece from Lanvin is a reminder that a little help does, in fact, go a long way."
- Subtitle: "Made to Scale" / Copy: "Gucci really 'weighed' some interesting decisions when making this vintage timepiece that gives whole new meaning to the idea of scaling up."
- Subtitle: "Gone Rotten" / Copy: "Gemstones may not expire, but artist Kathleen Ryan still found a way to make lapis lazuli, malachite, and turquoise look like mold in her rotting raspberry sculpture."

=== END EXAMPLES ===

STYLE:
- Subtitles: Catchy, clever, use wordplay/puns/pop culture references
- Copy: Punchy, specific, include names and dollar amounts
- AVOID: generic descriptions, corporate-speak"""

                gbu_prompt = f"""Write content for "The {section_key.split('_')[1].title()}" section.

Article: {article.get('title', '')}
Research: {article.get('research', article.get('snippet', ''))}
URL: {article.get('url', '')}

Requirements:
- Subtitle: Maximum 8 words, catchy and clever
- Copy: 1-2 sentences, maximum 30 words total
- Include SPECIFIC names, dollar amounts, locations, and details
- Tone: {section_type.get(section_key, 'engaging')}

Return JSON:
{{"subtitle": "...", "copy": "...", "url": "{article.get('url', '')}"}}"""

                try:
                    response = claude_client.generate_content(
                        prompt=gbu_prompt,
                        system_prompt=gbu_system,
                        max_tokens=200,
                        temperature=0.7
                    )

                    content = response.get('content', '{}')
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0].strip()
                    elif '```' in content:
                        content = content.split('```')[1].split('```')[0].strip()

                    generated[section_key] = json.loads(content)

                except Exception as e:
                    safe_print(f"  Error generating {section_key}: {e}")
                    generated[section_key] = {
                        'subtitle': article.get('title', '')[:50],
                        'copy': article.get('snippet', '')[:100],
                        'url': article.get('url', '')
                    }

        # Generate Industry Pulse (combined story from multiple articles)
        safe_print(f"[DEBUG] About to check Industry Pulse generation...")
        safe_print(f"[DEBUG] 'industry_pulse' in sections_data: {'industry_pulse' in sections_data}")

        # Always initialize industry_pulse to ensure it's in the response
        generated['industry_pulse'] = None

        if 'industry_pulse' in sections_data:
            safe_print(f"[DEBUG] Industry Pulse key found in sections_data")
            articles = sections_data['industry_pulse']
            safe_print(f"[DEBUG] articles value: {articles}")
            safe_print(f"[DEBUG] articles type: {type(articles)}")

            if not isinstance(articles, list):
                safe_print(f"[DEBUG] articles is not a list, converting...")
                articles = [articles] if articles else []
                safe_print(f"[DEBUG] after conversion: {articles}")

            # Only generate if we have articles selected
            safe_print(f"[DEBUG] Checking if should generate: not articles={not articles}, len(articles)={len(articles) if isinstance(articles, list) else 'N/A'}")

            if articles and len(articles) > 0:
                safe_print(f"  - Industry Pulse: Generating from {len(articles)} articles")
                safe_print(f"  - Article details: {[{k: v for k, v in a.items() if k in ['title', 'url']} for a in articles if isinstance(a, dict)]}")

                try:
                    articles_text = ""
                    for i, art in enumerate(articles):
                        articles_text += f"""
Article {i+1}: {art.get('title', '')}
Research: {art.get('research', art.get('snippet', ''))}
URL: {art.get('url', '')}
"""

                    pulse_system = """You write the "Industry Pulse" section for "Stay In The Loupe," a jewelry industry newsletter.

=== REAL EXAMPLES FROM PAST ISSUES (match this voice, depth, and specificity) ===

EXAMPLE 1 - Title: "Tariff Talk: The Latest Effects on the Jewelry Industry"
Intro: "As tariffs loom and restrictions are implemented on imported goods, the jewelry industry is taking action. In August, representatives with Jewelers of America headed to Washington, DC to speak with the Trump administration on the drastic effect this poses to the market."
H3 "Swiss Watches": "As of August 7, there is now a 39% tariff on Swiss imports, significantly affecting the luxury watch industry. In anticipation of the restriction, timepiece exports to the US surged 45% in July compared to the same time last year, according to JCK."
H3 "Diamond Disruption": "The new 50% tariffs on Indian goods, a jump from 25% previously, are projected to have a huge impact on the US diamond market. 'India is a dominant force in the global market for cut and polished diamonds, and America is one of its largest buyers,' says the New York Times."

EXAMPLE 2 - Title: "Not Your Grandmother's Brooch: The Fashion Statement Continues Making a Big Comeback"
Intro: "In November, news broke that a brooch once belonging to Napoleon Bonaparte netted a whopping $4.4 million at auction, far surpassing the $150,000-$200,000 price tag the 13-carat piece was expected to sell for. But, anyone paying attention to the latest trends may not have found the development that surprising — brooches are back in a big way this year."
H3 "Young Wearers Are Leading the Way": "According to Harper's Bazaar, 'the return of adornment is blending nostalgia, self-expression, and a touch of playful luxury in a market driven by individuality and emotion.' Millennials and Gen Z are at the forefront of this trend, with brooches seen on aspirational figures like Dua Lipa, Zendaya, and Gigi Hadid."

=== END EXAMPLES ===

WRITING STYLE:
- Include DIRECT QUOTES from named sources with titles (e.g., "JA President and CEO David J. Bonaparte shared...")
- Use specific dollar amounts, percentages, and statistics
- Be conversational -- use contractions, vary sentence length
- AVOID: "landscape", "navigate", "leverage", "robust", "in today's evolving"
"""

                    pulse_prompt = f"""Combine these jewelry industry articles into one cohesive "Industry Pulse" story.

Articles:
{articles_text}

Requirements:
- Title: Compelling, specific headline (max 15 words) -- descriptive, not generic
- Intro: 1-2 paragraphs setting up the story with specific facts, quotes, and data
- H3 Section 1: Descriptive heading (max 10 words) + 1-2 paragraphs with SPECIFIC data, DIRECT QUOTES, source references
- H3 Section 2: Descriptive heading (max 10 words) + 1-2 paragraphs with SPECIFIC data, DIRECT QUOTES, source references
- Include hyperlinks using markdown format [link text](URL)

Return JSON:
{{
    "title": "...",
    "intro": "...",
    "h3_1_title": "...",
    "h3_1_content": "...",
    "h3_2_title": "...",
    "h3_2_content": "...",
    "sources": [list of URLs used]
}}"""

                    response = claude_client.generate_content(
                        prompt=pulse_prompt,
                        system_prompt=pulse_system,
                        max_tokens=800,
                        temperature=0.6
                    )

                    content = response.get('content', '{}')
                    safe_print(f"  Industry Pulse raw response: {content[:200]}...")

                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0].strip()
                    elif '```' in content:
                        content = content.split('```')[1].split('```')[0].strip()

                    parsed = json.loads(content)
                    generated['industry_pulse'] = parsed
                    safe_print(f"  Industry Pulse generated successfully: {parsed.get('title', 'No title')}")

                except Exception as e:
                    safe_print(f"  ERROR generating industry_pulse: {e}")
                    import traceback
                    traceback.print_exc()
                    generated['industry_pulse'] = {
                        'title': 'Industry Pulse',
                        'intro': articles[0].get('snippet', '') if articles else '',
                        'h3_1_title': 'Key Developments',
                        'h3_1_content': '',
                        'h3_2_title': 'What This Means',
                        'h3_2_content': ''
                    }
            else:
                safe_print("  - Industry Pulse: No articles selected, skipping generation")

        # Generate Partner Advantage
        if 'partner_advantage' in sections_data and sections_data['partner_advantage']:
            articles = sections_data['partner_advantage']
            if not isinstance(articles, list):
                articles = [articles]

            articles_text = ""
            for art in articles:
                articles_text += f"- {art.get('title', '')}: {art.get('research', art.get('snippet', ''))}\n"

            partner_system = """You write the "Partner Advantage" section for "Stay In The Loupe," a jewelry industry newsletter. This section provides practical, actionable tips for jewelers based on industry articles.

=== REAL EXAMPLES FROM PAST ISSUES (match this practical, quote-rich voice) ===

EXAMPLE 1 - Subheader: "Promote Self-Gifting to Drive Pre-Holiday Sales"
Intro: "Who doesn't love buying themselves a gift? Especially when the timing is right. This fall, as summer vacation and back-to-school spending dies down and before people really start making holiday gift lists, it's the perfect time to start marketing self-purchases. 'While many jewelers write off this time of year as slow, there's a golden opportunity hiding in plain sight,' says National Jeweler."
Tips:
- "Look for signs." / "Train your staff to look for clues, which may be subtle. For example, comments like: 'This would look great on me,' or 'I've been eyeing this piece,' or 'I love this style.'"
- "Adjust your message." / "Promote the idea in marketing materials, your website, and social media. Use verbiage about 'rewarding yourself' and 'personal expression.'"
- "Segment your client base." / "Send emails to those who have purchased before to encourage the activity again."

EXAMPLE 2 - Subheader: "Stay Social: 4 Ways to Connect with Your Online Audience This Holiday Season"
Intro: "Holiday shoppers will be spending a lot of time online this season when curating their own gift guides. According to Google, consumers 'search online before 92% of store visits,' meaning meeting them in the digital space is imperative for sales."
Tips:
- "Be active and interactive." / "As the magazine suggests, 'Dedicate just a few minutes a day to follow new accounts, like posts, and leave thoughtful comments.' The payoff is not only building your community but also building trust."
- "Write back!" / "Social media is not the place to stay quiet. 'Users value direct, personal interactions,' says InStore, recommending to always have a friendly tone."
- "Show off real customers." / "Ask customers if they would be willing to take a photo in-store wearing their jewelry that you can then share online. 'These authentic images help humanize your brand and build emotional connections,' says InStore."

EXAMPLE 3 - Subheader: "Beat the Post-Holiday Blues: 5 Other Major Events You Should Market Around"
Intro: "Once the holidays are over this year, and sales trends start to take a dip with fewer gift purchases, there's no need to worry. There are other special events that you can effectively market to customers to fill in what National Jeweler calls the 'gifting gap.'"
Tips:
- "Anniversaries." / "If you sell engagement rings, you already have a base to follow up with. Offer a 'curated selection of gifts by year: gold for 1st, diamonds for 10th,' says the magazine."
- "New Babies." / "Parents love the chance to mark a new arrival to the family, so this is a great opportunity to market birthstone jewelry and charms."
- "Self Purchases." / "BriteCo data has found that 80% of people buy themselves jewelry rather than waiting for it as a gift. For marketing self-purchases, National Jeweler advises, 'Build marketing messaging that empowers your audience to celebrate their own achievements.'"

=== END EXAMPLES ===

WRITING STYLE:
- Include DIRECT QUOTES from articles (with attribution)
- Be conversational -- use contractions, address the reader as "you"
- Sound like a knowledgeable colleague sharing practical advice
- AVOID: "leverage", "navigate", "landscape", "robust", "in today's evolving"
"""

            partner_prompt = f"""Write a "Partner Advantage" section based on these articles:

{articles_text}

Requirements:
- Subheader: Max 15 words, catchy and specific
- Intro: 1 paragraph that hooks with a specific stat or quote, names the source, sets up the tips
- 3-5 tips, each with:
  - Mini-title: Short, punchy, action-oriented (max 10 words)
  - Supporting text: 1-3 sentences with DIRECT QUOTES from sources and specific data
- Include hyperlinks using markdown format [link text](URL)

Return JSON:
{{
    "subheader": "...",
    "intro": "...",
    "tips": [
        {{"title": "...", "content": "...", "url": "..."}},
        ...
    ]
}}"""

            try:
                response = claude_client.generate_content(
                    prompt=partner_prompt,
                    system_prompt=partner_system,
                    max_tokens=800,
                    temperature=0.6
                )

                content = response.get('content', '{}')
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()

                generated['partner_advantage'] = json.loads(content)

            except Exception as e:
                safe_print(f"  Error generating partner_advantage: {e}")
                generated['partner_advantage'] = {
                    'subheader': 'Tips for Success',
                    'intro': '',
                    'tips': []
                }

        # Generate Industry News (5 bullet points)
        if 'industry_news' in sections_data and sections_data['industry_news']:
            articles = sections_data['industry_news']
            if not isinstance(articles, list):
                articles = [articles]

            # Deduplicate articles by URL
            seen_urls = set()
            unique_articles = []
            for art in articles[:5]:
                url = art.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_articles.append(art)
                elif not url:
                    unique_articles.append(art)

            articles_text = ""
            for i, art in enumerate(unique_articles, 1):
                # Use research summary if available (from GPT-5.2), otherwise snippet
                detail = art.get('research', '') or art.get('snippet', art.get('content', ''))
                articles_text += f"ARTICLE {i}: \"{art.get('title', '')}\"\n  Source URL: {art.get('url', '')}\n  Key Details: {str(detail)[:400]}\n\n"

            news_system = """You write "Industry News" headline-style bullet points for "Stay In The Loupe," a jewelry industry newsletter.

=== REAL EXAMPLES (match this headline-capitalized style exactly) ===

"A 21.25-Carat Pink Diamond Worth $25 Million Was Stolen in Dubai, Part of an Elaborate Year-Long Scheme; It Has Since Been Recovered"

"US Jewelry Sales Are on the Rise, with CNBC Reporting an Increase of 5% in the First Half of 2025, Even As Economic Worries Loom"

"In a Total Rockstar Move, Elton John Turned His Old Kneecaps Into Jewelry; After They Were Surgically Removed, Pieces Were Used for a Necklace and Brooch"

"Could Gold Hit $4,000 an Ounce? All Signs Say Yes as It Hit Another Milestone in September, Selling for $3,800. Deutsche Bank Predicts $4,000 Will Happen by the End of the Year."

"Holiday Shopping Is Going to Set a New Record in 2025 with Analysts Predicting Sales Will Surpass the $1 Trillion Mark for the First Time Ever"

"The '90s Are Calling: Toe Rings Are Back in a Big Way This Year, But This Time It's All About Precious Metals, Pearls, and Big Carat Bling"

"Antique Diamonds Are Having a Moment Thanks to the Taylor Swift Effect, with Many Engagement Ring Shoppers Now Looking for Mine-Cut Stones"

=== END EXAMPLES ===

STYLE:
- HEADLINE CAPITALIZED (Title Case Throughout)
- Include SPECIFIC dollar amounts, percentages, names, and details -- never be vague
- Attention-grabbing and vivid -- use pop culture references, punchy semicolons, and dramatic phrasing
- Each bullet: ONE sentence, 15-30 words"""

            news_prompt = f"""Write exactly {len(unique_articles)} DIFFERENT "Industry News" bullet points — one per article below.

{articles_text}

CRITICAL RULES:
1. Bullet 1 MUST be about ARTICLE 1's specific topic: "{unique_articles[0].get('title', '')}" — and ONLY that topic
2. Bullet 2 MUST be about ARTICLE 2's specific topic: "{unique_articles[1].get('title', '') if len(unique_articles) > 1 else ''}" — and ONLY that topic
3. Bullet 3 MUST be about ARTICLE 3's specific topic: "{unique_articles[2].get('title', '') if len(unique_articles) > 2 else ''}" — and ONLY that topic
4. Bullet 4 MUST be about ARTICLE 4's specific topic: "{unique_articles[3].get('title', '') if len(unique_articles) > 3 else ''}" — and ONLY that topic
5. Bullet 5 MUST be about ARTICLE 5's specific topic: "{unique_articles[4].get('title', '') if len(unique_articles) > 4 else ''}" — and ONLY that topic

Return JSON only:
{{
    "bullets": [
        {{"text": "Headline-Style Sentence About Article 1 Topic with Specific Details", "url": "article1_url"}},
        {{"text": "Different Headline-Style Sentence About Article 2 Topic", "url": "article2_url"}},
        ...
    ]
}}
"""

            try:
                response = claude_client.generate_content(
                    prompt=news_prompt,
                    system_prompt=news_system,
                    max_tokens=600,
                    temperature=0.8
                )

                content = response.get('content', '{}')
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()

                generated['industry_news'] = json.loads(content)

            except Exception as e:
                safe_print(f"  Error generating industry_news: {e}")
                # Fallback: use article titles as link text in a sentence
                generated['industry_news'] = {
                    'bullets': [{'text': f"[{art.get('title', '')}]({art.get('url', '')}) offers insights for jewelry professionals.", 'url': art.get('url', '')} for art in articles[:5]]
                }

        # Auto-generate intro if not provided
        if not intro_content:
            safe_print("  - Auto-generating introduction...")
            try:
                # Build context from generated sections
                sections_context = []
                if generated.get('the_good'):
                    sections_context.append(f"positive news: {generated['the_good'].get('subtitle', '')}")
                if generated.get('industry_pulse'):
                    sections_context.append(f"industry pulse: {generated['industry_pulse'].get('title', '')}")
                if generated.get('partner_advantage'):
                    sections_context.append(f"tips: {generated['partner_advantage'].get('subheader', '')}")

                context_text = ", ".join(sections_context[:3]) if sections_context else "jewelry industry updates"

                intro_system = """You write newsletter introductions for "Stay In The Loupe," a jewelry industry newsletter.

=== REAL EXAMPLES (match this conversational voice) ===
- "You asked, and BriteCo delivers: Now you can choose from four different appraisal designs to create the look that suits you best. Plus, keep reading to discover findings from our lab-grown vs. natural diamond report, get the latest on tariffs affecting the industry, and find tips on promoting self-purchases."
- "The holidays are coming, and we're unwrapping the biggest trends to watch out for this season -- including how to gift yourself with prominent sales opportunities. Plus, we take a closer look at the benefits of Google Ads vs. social media ads and when to utilize each one."
- "Did you know BriteCo has a thriving blog and YouTube channel? As we continue to ramp up content, we are looking for expert voices that can share advice on a range of jewelry topics."
=== END EXAMPLES ===

VOICE:
- Warm and conversational
- Use contractions (we're, you'll, don't)
- Tease 2-3 specific topics from this issue
- 2-3 sentences max, under 50 words
- AVOID: "Welcome to", "In this issue", generic openings"""

                intro_prompt = f"""Write a brief introduction for the {month.capitalize()} edition.

This issue covers: {context_text}

Output ONLY the intro text, 2-3 sentences."""

                intro_result = claude_client.generate_content(
                    prompt=intro_prompt,
                    system_prompt=intro_system,
                    temperature=0.7,
                    max_tokens=150
                )
                generated['intro'] = intro_result['content'].strip()
                safe_print(f"    Intro generated: {len(generated['intro'].split())} words")
            except Exception as e:
                safe_print(f"    Error auto-generating intro: {e}")
                generated['intro'] = f"Welcome to the {month.capitalize()} edition of Stay In The Loupe!"
        else:
            generated['intro'] = intro_content

        # Auto-generate brite spot if not provided
        if not brite_spot_content:
            safe_print("  - Auto-generating Brite Spot...")
            try:
                brite_spot_system = """You write "The Brite Spot" section for "Stay In The Loupe," a jewelry newsletter.

=== REAL EXAMPLES (match this warm, direct voice) ===
- "We are wishing you a very happy holiday season and thank you for another great year of partnership. We value your contributions towards our mission of helping clients protect their jewelry."
- "We want to prioritize the next wave of POS integrations and need your help to do so. Please fill out our quick and easy form and tell us your preferred point-of-sale system. When you do, you'll be entered to win a $50 gift card!"
- "BriteCo is building up a database of expert voices to include in our blog and YouTube channel. Every time you're interviewed, we'll give you $20 and your name and website will be featured."
=== END EXAMPLES ===

VOICE:
- Warm, genuine, not corporate
- Focus on partnership value
- Use "we" and "you"
- Under 100 words
- Include a subtle call to action"""

                brite_spot_prompt = f"""Write a "Brite Spot" section for the {month.capitalize()} edition.

Topic: Thank partners for their continued partnership, mention any relevant updates or ways they can engage with BriteCo.

Output JSON with "title" (max 10 words) and "body" (max 100 words):
{{"title": "...", "body": "..."}}"""

                brite_result = claude_client.generate_content(
                    prompt=brite_spot_prompt,
                    system_prompt=brite_spot_system,
                    temperature=0.6,
                    max_tokens=200
                )
                content_str = brite_result['content'].strip()
                if '```json' in content_str:
                    content_str = content_str.split('```json')[1].split('```')[0].strip()
                elif '```' in content_str:
                    content_str = content_str.split('```')[1].split('```')[0].strip()
                brite_json = json.loads(content_str)
                generated['brite_spot'] = brite_json
                safe_print(f"    Brite Spot generated: {brite_json.get('title', 'No title')}")
            except Exception as e:
                safe_print(f"    Error auto-generating brite spot: {e}")
                generated['brite_spot'] = {
                    "title": "Thank You for Your Partnership",
                    "body": "We appreciate your continued partnership and look forward to supporting your jewelry business."
                }
        else:
            generated['brite_spot'] = brite_spot_content

        # Convert markdown links to HTML in all generated content
        generated = process_generated_content(generated)

        return jsonify({
            'success': True,
            'generated': generated,
            'month': month
        })

    except Exception as e:
        safe_print(f"[API ERROR] Generate newsletter: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTES - BRAND CHECK
# ============================================================================

@app.route('/api/check-brand-guidelines', methods=['POST'])
def check_brand_guidelines():
    """Check content against brand guidelines"""
    try:
        data = request.json
        content = data.get('content', {})
        month = data.get('month', '')

        safe_print(f"\n[API] Checking brand guidelines for {month}...")

        # Build content summary
        content_text = json.dumps(content, indent=2)

        prompt = f"""Review this jewelry newsletter content against brand guidelines.

Content:
{content_text[:4000]}

BRAND GUIDELINES FOR "STAY IN THE LOUPE" NEWSLETTER:

1. VOICE & TONE:
- Professional but personable - like a well-connected colleague sharing the latest scoop
- Industry-savvy and knowledgeable about jewelry trends, materials, and market dynamics
- Warm and helpful - uses "we" and "you" frequently
- Playful but professional - incorporates wordplay and light humor without being corny

2. CONTENT RESTRICTIONS:
- EXCLUDE: Personnel announcements, deaths/obituaries, political content
- EXCLUDE: Non-jewelry topics, overly salesy language
- Word limits: Good/Bad/Ugly copy max 30 words each, paragraphs max 60 words

3. PUNCTUATION & FORMATTING:
- Use serial comma in lists (red, white, and blue)
- Use em dash (—) with spaces around it
- Put punctuation inside quotation marks

4. BRITECO BRAND TERMINOLOGY:
- DO: Call BriteCo an "insurtech company" or "insurance provider"
- DO: Say "backed by an AM Best A+ rated Insurance Carrier"
- DON'T: Call BriteCo an "insurance company"
- DON'T: Say "we have AM Best policies" or "we are AM Best"

Review the content and identify SPECIFIC phrases that need to be changed.

IMPORTANT: Skip over hyperlinks and URLs - do not flag them as issues. Hyperlinks in formats like [text](url) or <a href="...">text</a> should be left as-is.

Return JSON:
{{
    "suggestions": [
        {{
            "section": "intro" | "brite_spot" | "the_good" | "the_bad" | "the_ugly" | "industry_pulse" | "partner_advantage" | "industry_news",
            "issue": "Brief description of the issue",
            "original": "exact phrase from content that needs changing",
            "suggested": "what it should be changed to",
            "reason": "why this change is needed per brand guidelines"
        }}
    ]
}}

Only include items that actually need to be changed. If the content is perfect, return an empty suggestions array."""

        response = claude_client.generate_content(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.2
        )

        result_content = response.get('content', '{}')
        if '```json' in result_content:
            result_content = result_content.split('```json')[1].split('```')[0].strip()
        elif '```' in result_content:
            result_content = result_content.split('```')[1].split('```')[0].strip()

        try:
            check_results = json.loads(result_content)
        except json.JSONDecodeError as e:
            safe_print(f"[API WARNING] Failed to parse brand check JSON: {e}")
            check_results = {"suggestions": []}

        num_suggestions = len(check_results.get('suggestions', []))
        passed = num_suggestions == 0

        safe_print(f"[API] Brand check complete - {num_suggestions} suggestions found")

        return jsonify({
            'success': True,
            'passed': passed,
            'check_results': check_results
        })

    except Exception as e:
        safe_print(f"[API ERROR] Brand check: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTES - IMAGE GENERATION
# ============================================================================

@app.route('/api/generate-image-prompts', methods=['POST'])
def generate_image_prompts():
    """Generate image prompts for newsletter sections"""
    try:
        data = request.json
        sections = data.get('sections', {})

        safe_print(f"\n[API] Generating image prompts for {len(sections)} sections...")

        prompts = {}

        for section, content in sections.items():
            if not content:
                continue

            safe_print(f"  - Creating image prompt for {section}")

            title = content.get('title', content.get('subtitle', ''))
            body = content.get('content', content.get('copy', content.get('intro', '')))

            # Handle case where body is a dict/object instead of string
            if isinstance(body, dict):
                body = body.get('intro', '') or body.get('content', '') or body.get('copy', '') or str(body)

            # Ensure body is a string before slicing
            body = str(body) if body else ''

            prompt_request = f"""Create a text-to-image prompt for a jewelry newsletter image.

Section: {section}
Title: "{title}"
Content: "{body[:400]}..."

Requirements:
- Photorealistic, professional photography style (NOT cartoon, NOT illustration, NOT digital art)
- Stock photo aesthetic - like images from Shutterstock or Getty Images
- Luxury jewelry aesthetic with elegant lighting and composition
- Teal/gold color accents where appropriate (BriteCo brand colors)
- No text overlays in the image
- Suitable for professional email newsletter
- Clean, well-lit, high-quality photography look
- Focus on jewelry, gemstones, watches, or luxury retail settings

Output ONLY the image generation prompt, nothing else."""

            try:
                response = claude_client.generate_content(
                    prompt=prompt_request,
                    model="claude-opus-4-5-20251101",
                    max_tokens=150,
                    temperature=0.5
                )

                prompts[section] = {
                    'prompt': response.get('content', '').strip(),
                    'title': title
                }

            except Exception as e:
                safe_print(f"  Error generating prompt for {section}: {e}")
                prompts[section] = {
                    'prompt': f"Photorealistic professional jewelry photography, elegant luxury display with soft lighting, high-end retail aesthetic, {title}, stock photo quality",
                    'title': title
                }

        safe_print(f"[API] Generated {len(prompts)} image prompts")

        return jsonify({
            'success': True,
            'prompts': prompts,
            'generated_at': datetime.now().isoformat()
        })

    except Exception as e:
        safe_print(f"[API ERROR] Image prompts: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/enhance-image-prompt', methods=['POST'])
def enhance_image_prompt():
    """Enhance a rough image description into a detailed, optimized image generation prompt"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt required'}), 400

        if not claude_client:
            return jsonify({'success': False, 'error': 'Claude client not available'}), 500

        system_prompt = """You are an expert image prompt engineer. Take the user's rough description and transform it into a highly detailed, optimized prompt for AI image generation.

RULES:
- Be specific about composition, lighting, color palette, and style
- Specify "professional photograph" or "digital illustration" style
- Include mood and atmosphere details
- Add details about perspective and framing
- NEVER include text, words, letters, or watermarks in the image description
- Keep the enhanced prompt under 200 words
- Output ONLY the enhanced prompt, nothing else"""

        result = claude_client.generate_content(
            prompt=f"Enhance this image prompt for newsletter use:\n\n{prompt}",
            system_prompt=system_prompt,
            max_tokens=300,
            temperature=0.7
        )

        enhanced = result.get('content', '').strip()

        return jsonify({
            'success': True,
            'enhanced_prompt': enhanced,
            'original_prompt': prompt
        })

    except Exception as e:
        safe_print(f"[API ERROR] Enhance prompt: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate-images', methods=['POST'])
def generate_images():
    """Generate images using Gemini"""
    try:
        data = request.json

        # Check if Gemini client is available
        if not gemini_client or not gemini_client.is_available():
            safe_print("[API ERROR] Gemini client not available")
            return jsonify({'success': False, 'error': 'Gemini image generation not available'}), 500

        # Handle single-image request (from special section)
        single_prompt = data.get('prompt')
        single_section = data.get('section')
        if single_prompt and single_section:
            safe_print(f"\n[API] Single image request for {single_section}")
            try:
                result = gemini_client.generate_image(
                    prompt=single_prompt,
                    aspect_ratio="16:9"
                )
                image_data = result.get('image_base64', result.get('image_data', ''))
                if image_data:
                    return jsonify({
                        'success': True,
                        'image_data': image_data,
                        'images': {single_section: f"data:image/png;base64,{image_data}"}
                    })
                else:
                    return jsonify({'success': False, 'error': 'No image data returned'})
            except Exception as e:
                safe_print(f"  [ERROR] Single image generation failed: {e}")
                return jsonify({'success': False, 'error': str(e)})

        prompts = data.get('prompts', {})

        safe_print(f"\n[API] Generating {len(prompts)} images...")

        images = {}

        # Image sizes for different sections
        # Square (350x350) for centered GBU images
        # Landscape (570x320) for Brite Spot (50% shorter than square at full width)
        # Full width (490x263) for: Industry Pulse, Partner Advantage
        IMAGE_SIZES = {
            'the_good': (350, 350),
            'the_bad': (350, 350),
            'the_ugly': (350, 350),
            'brite_spot': (570, 320),
            'industry_pulse': (490, 263),
            'partner_advantage': (490, 263)
        }

        # Aspect ratios: square for GBU, landscape for full-width sections
        ASPECT_RATIOS = {
            'the_good': '1:1',
            'the_bad': '1:1',
            'the_ugly': '1:1',
            'brite_spot': '16:9',
            'industry_pulse': '16:9',
            'partner_advantage': '16:9'
        }

        for section, prompt_data in prompts.items():
            prompt = prompt_data.get('prompt', '')
            if not prompt:
                continue

            safe_print(f"  Generating image for {section}...")

            try:
                aspect_ratio = ASPECT_RATIOS.get(section, '1:1')
                # Use default model (gemini-2.5-flash-image) from gemini_client
                result = gemini_client.generate_image(
                    prompt=prompt,
                    aspect_ratio=aspect_ratio
                )

                image_data = result.get('image_data', '')

                # Resize to target dimensions
                if image_data:
                    target_size = IMAGE_SIZES.get(section, (180, 180))
                    image_data = resize_image(image_data, target_size)

                images[section] = {
                    'url': f"data:image/png;base64,{image_data}" if image_data else '',
                    'prompt': prompt
                }
                safe_print(f"    Generated image for {section}")

            except Exception as e:
                safe_print(f"  Error generating image for {section}: {e}")
                import traceback
                traceback.print_exc()
                images[section] = {
                    'url': '',
                    'prompt': prompt,
                    'error': str(e)
                }

        return jsonify({
            'success': True,
            'images': images
        })

    except Exception as e:
        safe_print(f"[API ERROR] Generate images: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def resize_image(base64_data, target_size):
    """Resize image to target dimensions"""
    try:
        from PIL import Image, ImageOps

        image_bytes = base64.b64decode(base64_data)
        pil_image = Image.open(BytesIO(image_bytes))

        # Use fit to maintain aspect ratio
        resized = ImageOps.fit(pil_image, target_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

        # Convert back to base64
        buffer = BytesIO()
        resized.save(buffer, format='PNG', optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        safe_print(f"  Resize error: {e}")
        return base64_data


@app.route('/api/generate-single-image', methods=['POST'])
def generate_single_image():
    """Generate a single image using Gemini"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        section = data.get('section', 'generic')

        if not prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'}), 400

        # Check if Gemini client is available
        if not gemini_client or not gemini_client.is_available():
            return jsonify({'success': False, 'error': 'Gemini image generation not available'}), 500

        safe_print(f"\n[API] Regenerating image for {section}...")

        # Image sizes for different sections
        IMAGE_SIZES = {
            'the_good': (350, 350),
            'the_bad': (350, 350),
            'the_ugly': (350, 350),
            'brite_spot': (570, 320),
            'industry_pulse': (490, 263),
            'partner_advantage': (490, 263)
        }

        # Aspect ratios
        ASPECT_RATIOS = {
            'the_good': '1:1',
            'the_bad': '1:1',
            'the_ugly': '1:1',
            'brite_spot': '16:9',
            'industry_pulse': '16:9',
            'partner_advantage': '16:9'
        }

        aspect_ratio = ASPECT_RATIOS.get(section, '1:1')
        result = gemini_client.generate_image(
            prompt=prompt,
            aspect_ratio=aspect_ratio
        )

        image_data = result.get('image_data', '')

        # Resize to target dimensions
        if image_data:
            target_size = IMAGE_SIZES.get(section, (180, 180))
            image_data = resize_image(image_data, target_size)

        image_url = f"data:image/png;base64,{image_data}" if image_data else ''

        return jsonify({
            'success': True,
            'image_url': image_url
        })

    except Exception as e:
        safe_print(f"[API ERROR] Generate single image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/resize-image', methods=['POST'])
def resize_image_endpoint():
    """Resize an image to specified dimensions"""
    try:
        data = request.json
        image_url = data.get('image_url', '')
        width = data.get('width', 203)
        height = data.get('height', 203)
        section = data.get('section', 'generic')

        if not image_url:
            return jsonify({'success': False, 'error': 'No image URL provided'}), 400

        safe_print(f"\n[API] Resizing image for {section} to {width}x{height}...")

        # Extract base64 data from data URL
        if image_url.startswith('data:'):
            base64_data = image_url.split(',')[1] if ',' in image_url else ''
        else:
            # If it's a regular URL, fetch and convert
            import requests as req
            response = req.get(image_url)
            base64_data = base64.b64encode(response.content).decode('utf-8')

        if not base64_data:
            return jsonify({'success': False, 'error': 'Could not extract image data'}), 400

        # Resize the image
        resized_data = resize_image(base64_data, (width, height))
        resized_url = f"data:image/png;base64,{resized_data}"

        return jsonify({
            'success': True,
            'resized_url': resized_url
        })

    except Exception as e:
        safe_print(f"[API ERROR] Resize image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTES - SUBJECT LINE
# ============================================================================

@app.route('/api/generate-subject-lines', methods=['POST'])
def generate_subject_lines():
    """Generate subject line and preheader options"""
    try:
        data = request.json
        content = data.get('content', {})
        month = data.get('month', '')
        tone = data.get('tone', 'professional')

        safe_print(f"\n[API] Generating subject lines for {month}, tone: {tone}...")

        content_summary = json.dumps(content, indent=2)[:1500]

        tone_descriptions = {
            'professional': 'Professional and informative',
            'friendly': 'Friendly and conversational',
            'urgent': 'Urgent and action-oriented',
            'playful': 'Playful and fun',
            'exclusive': 'Exclusive and premium'
        }

        prompt = f"""Generate 5 subject lines and 5 preheaders for a jewelry newsletter.

Month: {month}
Tone: {tone_descriptions.get(tone, 'Professional')}

Content Summary:
{content_summary}

Requirements:
- Subject lines: 40-60 characters, compelling, may use questions or numbers
- Preheaders: 80-100 characters, complement subject line

Return JSON:
{{
    "subject_lines": ["...", "...", "...", "...", "..."],
    "preheaders": ["...", "...", "...", "...", "..."]
}}"""

        response = claude_client.generate_content(
            prompt=prompt,
            max_tokens=500,
            temperature=0.8
        )

        result_content = response.get('content', '{}')
        if '```json' in result_content:
            result_content = result_content.split('```json')[1].split('```')[0].strip()
        elif '```' in result_content:
            result_content = result_content.split('```')[1].split('```')[0].strip()

        options = json.loads(result_content)

        return jsonify({
            'success': True,
            'subject_lines': options.get('subject_lines', []),
            'preheaders': options.get('preheaders', [])
        })

    except Exception as e:
        safe_print(f"[API ERROR] Subject lines: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTES - EXPORT & EMAIL
# ============================================================================

@app.route('/api/send-preview', methods=['POST'])
def send_preview():
    """Send newsletter preview to team members via SendGrid"""
    try:
        data = request.json
        recipients = data.get('recipients', [])
        subject = data.get('subject', 'Stay In The Loupe Preview')
        html_content = data.get('html', '')

        if not recipients or not html_content:
            return jsonify({"success": False, "error": "Recipients and HTML content required"}), 400

        safe_print(f"[API] Sending preview to {len(recipients)} recipients via SendGrid...")

        if not SENDGRID_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "SendGrid library not installed. Run: pip install sendgrid"
            }), 500

        sendgrid_api_key = os.environ.get('SENDGRID_API_KEY') or os.environ.get('_SENDGRID_API_KEY')
        from_email = os.environ.get('SENDGRID_FROM_EMAIL') or os.environ.get('_SENDGRID_FROM_EMAIL') or 'jeweler@brite.co'
        from_name = os.environ.get('SENDGRID_FROM_NAME') or os.environ.get('_SENDGRID_FROM_NAME') or 'Stay In The Loupe'

        if not sendgrid_api_key:
            return jsonify({
                "success": False,
                "error": "SendGrid API key not configured."
            }), 500

        sg = sendgrid.SendGridAPIClient(api_key=sendgrid_api_key)

        sent_count = 0
        errors = []

        for recipient in recipients:
            try:
                message = Mail(
                    from_email=(from_email, from_name),
                    to_emails=recipient,
                    subject=subject,
                    html_content=html_content
                )

                response = sg.send(message)

                if response.status_code in [200, 201, 202]:
                    sent_count += 1
                else:
                    errors.append(f"Failed for {recipient}: status {response.status_code}")

            except Exception as email_error:
                errors.append(f"Failed for {recipient}: {str(email_error)}")

        return jsonify({
            "success": sent_count > 0,
            "message": f"Preview sent to {sent_count} recipient(s)",
            "errors": errors if errors else None
        })

    except Exception as e:
        safe_print(f"[API] Send preview error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/export-to-docs', methods=['POST'])
def export_to_docs():
    """Export newsletter content to Google Docs"""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        data = request.json
        content = data.get('content', {})
        month = data.get('month', datetime.now().strftime('%B'))
        year = data.get('year', datetime.now().year)
        # Format: "2026 January - Jeweler Newsletter"
        title = data.get('title', f"{year} {month} - Jeweler Newsletter")
        send_email = data.get('send_email', False)
        recipients = data.get('recipients', [])

        safe_print(f"[API] Exporting to Google Docs: {title}")

        creds_json = os.environ.get('GOOGLE_DOCS_CREDENTIALS') or os.environ.get('_GOOGLE_DOCS_CREDENTIALS')

        if not creds_json:
            return jsonify({
                "success": False,
                "error": "Google Docs credentials not configured."
            }), 500

        try:
            creds_data = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(
                creds_data,
                scopes=['https://www.googleapis.com/auth/documents', 'https://www.googleapis.com/auth/drive']
            )
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Invalid Google credentials: {str(e)}"
            }), 500

        docs_service = build('docs', 'v1', credentials=credentials)
        drive_service = build('drive', 'v3', credentials=credentials)

        # Create doc in the specified folder
        file_metadata = {
            'name': title,
            'mimeType': 'application/vnd.google-apps.document',
            'parents': [GOOGLE_DRIVE_FOLDER_ID]
        }

        created_file = drive_service.files().create(
            body=file_metadata,
            fields='id',
            supportsAllDrives=True
        ).execute()

        doc_id = created_file.get('id')
        doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"

        safe_print(f"[API] Created Google Doc: {doc_id}")

        # Build document content
        requests_list = []
        index_offset = [1]

        def add_text(text, bold=False, heading=False, link_url=None):
            if not text:
                return
            text = str(text).strip() + '\n\n'
            start_index = index_offset[0]
            end_index = start_index + len(text)

            requests_list.append({
                'insertText': {
                    'location': {'index': start_index},
                    'text': text
                }
            })

            if heading:
                requests_list.append({
                    'updateParagraphStyle': {
                        'range': {'startIndex': start_index, 'endIndex': end_index - 1},
                        'paragraphStyle': {'namedStyleType': 'HEADING_2'},
                        'fields': 'namedStyleType'
                    }
                })
            elif bold:
                requests_list.append({
                    'updateTextStyle': {
                        'range': {'startIndex': start_index, 'endIndex': end_index - 1},
                        'textStyle': {'bold': True},
                        'fields': 'bold'
                    }
                })

            # Add hyperlink if URL provided
            if link_url:
                requests_list.append({
                    'updateTextStyle': {
                        'range': {'startIndex': start_index, 'endIndex': end_index - 2},  # -2 to exclude \n\n
                        'textStyle': {
                            'link': {'url': link_url},
                            'foregroundColor': {'color': {'rgbColor': {'red': 0.0, 'green': 0.51, 'blue': 0.51}}}
                        },
                        'fields': 'link,foregroundColor'
                    }
                })

            index_offset[0] = end_index

        # Add content sections
        add_text(title, heading=True)

        if content.get('intro'):
            add_text(content['intro'])

        if content.get('brite_spot'):
            add_text('The Brite Spot', bold=True)
            add_text(content['brite_spot'])

        for section in ['the_good', 'the_bad', 'the_ugly']:
            if content.get(section):
                section_title = section.replace('_', ' ').title()
                add_text(section_title, bold=True)
                sec = content[section]
                if isinstance(sec, dict):
                    add_text(f"{sec.get('subtitle', '')}\n{sec.get('copy', '')}")
                else:
                    add_text(str(sec))

        if content.get('industry_pulse'):
            add_text('Industry Pulse', bold=True)
            pulse = content['industry_pulse']
            if isinstance(pulse, dict):
                add_text(pulse.get('title', ''))
                add_text(pulse.get('intro', ''))
                add_text(pulse.get('h3_1_title', ''), bold=True)
                add_text(pulse.get('h3_1_content', ''))
                add_text(pulse.get('h3_2_title', ''), bold=True)
                add_text(pulse.get('h3_2_content', ''))
            else:
                add_text(str(pulse))

        if content.get('partner_advantage'):
            add_text('Partner Advantage', bold=True)
            pa = content['partner_advantage']
            if isinstance(pa, dict):
                add_text(pa.get('subheader', ''))
                add_text(pa.get('intro', ''))
                for tip in pa.get('tips', []):
                    if isinstance(tip, dict):
                        add_text(f"• {tip.get('title', '')}: {tip.get('content', '')}")
                    else:
                        add_text(f"• {tip}")
            else:
                add_text(str(pa))

        if content.get('industry_news'):
            add_text('Industry News', bold=True)
            news = content['industry_news']
            if isinstance(news, dict) and 'bullets' in news:
                for bullet in news['bullets']:
                    if isinstance(bullet, dict):
                        bullet_text = f"• {bullet.get('text', '')}"
                        bullet_url = bullet.get('url', None)
                        add_text(bullet_text, link_url=bullet_url)
                    else:
                        add_text(f"• {bullet}")
            elif isinstance(news, list):
                for item in news:
                    add_text(f"• {item}")

        # Special Section (if included)
        if content.get('special_section'):
            ss = content['special_section']
            ss_title = ss.get('title', 'Special Section') if isinstance(ss, dict) else 'Special Section'
            ss_body = ss.get('body', str(ss)) if isinstance(ss, dict) else str(ss)
            add_text(ss_title, bold=True)
            add_text(ss_body)

        # Execute batch update
        if requests_list:
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': requests_list}
            ).execute()

        # Make accessible
        drive_service.permissions().create(
            fileId=doc_id,
            body={'type': 'anyone', 'role': 'reader'},
            supportsAllDrives=True
        ).execute()

        # Send email if requested
        emails_sent = []
        if send_email and recipients and SENDGRID_AVAILABLE:
            sendgrid_api_key = os.environ.get('SENDGRID_API_KEY') or os.environ.get('_SENDGRID_API_KEY')
            from_email = os.environ.get('SENDGRID_FROM_EMAIL') or 'jeweler@brite.co'
            from_name = os.environ.get('SENDGRID_FROM_NAME') or 'Stay In The Loupe'

            if sendgrid_api_key:
                sg = sendgrid.SendGridAPIClient(api_key=sendgrid_api_key)

                for recipient in recipients:
                    try:
                        email_html = f"""
                        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                            <h2 style="color: #008181;">Stay In The Loupe - Ready for Review</h2>
                            <p>The <strong>{month} {year}</strong> newsletter has been exported to Google Docs.</p>
                            <p style="margin: 20px 0;">
                                <a href="{doc_url}" style="background: #008181; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px;">
                                    Open Google Doc
                                </a>
                            </p>
                        </div>
                        """

                        message = Mail(
                            from_email=(from_email, from_name),
                            to_emails=recipient,
                            subject=f"Stay In The Loupe - {month} {year} - Ready for Review",
                            html_content=email_html
                        )

                        response = sg.send(message)
                        if response.status_code in [200, 201, 202]:
                            emails_sent.append(recipient)

                    except Exception as email_error:
                        safe_print(f"[API] Email error for {recipient}: {email_error}")

        return jsonify({
            "success": True,
            "doc_url": doc_url,
            "doc_id": doc_id,
            "title": title,
            "emails_sent": emails_sent
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# ROUTES - CUSTOM LINK & DOC NOTIFICATION
# ============================================================================

@app.route('/api/fetch-article-metadata', methods=['POST'])
def fetch_article_metadata():
    """Fetch metadata from a custom article URL"""
    try:
        import requests as req
        from bs4 import BeautifulSoup

        data = request.json
        url = data.get('url', '')

        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400

        safe_print(f"\n[API] Fetching metadata for: {url}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            response = req.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except req.RequestException as e:
            return jsonify({
                'success': False,
                'error': f'Failed to fetch URL: {str(e)}'
            }), 400

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title = ''
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            title = og_title['content']
        elif soup.title:
            title = soup.title.string or ''

        # Extract description
        description = ''
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            description = og_desc['content']
        else:
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                description = meta_desc['content']

        # Extract image
        image = ''
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            image = og_image['content']

        # Extract publisher/site name
        publisher = extract_domain(url)
        og_site = soup.find('meta', property='og:site_name')
        if og_site and og_site.get('content'):
            publisher = og_site['content']

        # Extract publish date
        published_date = ''
        article_date = soup.find('meta', property='article:published_time')
        if article_date and article_date.get('content'):
            published_date = article_date['content']

        return jsonify({
            'success': True,
            'metadata': {
                'title': title.strip() if title else 'Untitled Article',
                'description': description.strip()[:500] if description else '',
                'image': image,
                'publisher': publisher,
                'published_date': published_date,
                'url': url
            }
        })

    except Exception as e:
        safe_print(f"[API ERROR] Fetch metadata: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/send-doc-notification', methods=['POST'])
def send_doc_notification():
    """Send email notification with Google Doc link"""
    try:
        data = request.json
        recipients = data.get('recipients', [])
        doc_url = data.get('doc_url', '')
        month = data.get('month', '')
        year = data.get('year', datetime.now().year)

        if not recipients or not doc_url:
            return jsonify({
                'success': False,
                'error': 'Recipients and doc_url are required'
            }), 400

        safe_print(f"\n[API] Sending doc notification to {len(recipients)} recipients...")

        if not SENDGRID_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'SendGrid library not installed. Run: pip install sendgrid'
            }), 500

        sendgrid_api_key = os.environ.get('SENDGRID_API_KEY') or os.environ.get('_SENDGRID_API_KEY')
        from_email = os.environ.get('SENDGRID_FROM_EMAIL') or os.environ.get('_SENDGRID_FROM_EMAIL') or 'jeweler@brite.co'
        from_name = os.environ.get('SENDGRID_FROM_NAME') or os.environ.get('_SENDGRID_FROM_NAME') or 'Stay In The Loupe'

        if not sendgrid_api_key:
            return jsonify({
                'success': False,
                'error': 'SendGrid API key not configured.'
            }), 500

        sg = sendgrid.SendGridAPIClient(api_key=sendgrid_api_key)

        sent_count = 0
        errors = []

        email_html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="color: #272d3f; margin: 0;">Stay In The Loupe</h1>
                <p style="color: #FE8916; margin: 5px 0;">Jeweler Newsletter</p>
            </div>

            <div style="background: #f5f7fa; border-radius: 8px; padding: 30px; text-align: center;">
                <h2 style="color: #272d3f; margin-top: 0;">Newsletter Ready for Review</h2>
                <p style="color: #5D7283; font-size: 16px;">
                    The <strong>{month} {year}</strong> edition of Stay In The Loupe has been exported to Google Docs and is ready for your review.
                </p>
                <p style="margin: 30px 0;">
                    <a href="{doc_url}" style="background: #008181; color: white; padding: 14px 32px; text-decoration: none; border-radius: 6px; font-weight: 600; display: inline-block;">
                        Open in Google Docs
                    </a>
                </p>
                <p style="color: #888; font-size: 14px;">
                    Click the button above to review and make any necessary edits.
                </p>
            </div>

            <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee;">
                <p style="color: #888; font-size: 12px; margin: 0;">
                    This notification was sent by the Stay In The Loupe newsletter generator.
                </p>
            </div>
        </div>
        """

        for recipient in recipients:
            try:
                message = Mail(
                    from_email=(from_email, from_name),
                    to_emails=recipient,
                    subject=f"Stay In The Loupe - {month} {year} - Ready for Review",
                    html_content=email_html
                )

                response = sg.send(message)

                if response.status_code in [200, 201, 202]:
                    sent_count += 1
                    safe_print(f"  Sent to: {recipient}")
                else:
                    errors.append(f"Failed for {recipient}: status {response.status_code}")

            except Exception as email_error:
                errors.append(f"Failed for {recipient}: {str(email_error)}")

        return jsonify({
            'success': sent_count > 0,
            'message': f"Notification sent to {sent_count} recipient(s)",
            'sent_count': sent_count,
            'errors': errors if errors else None
        })

    except Exception as e:
        safe_print(f"[API ERROR] Send doc notification: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTES - ONTRAPORT
# ============================================================================

@app.route('/api/push-to-ontraport', methods=['POST'])
def push_to_ontraport():
    """Push newsletter to Ontraport"""
    try:
        import requests as req

        data = request.json
        subject = data.get('subject')
        month = data.get('month', 'Newsletter')
        html_content = data.get('html')

        if not subject or not html_content:
            return jsonify({'success': False, 'error': 'Missing subject or HTML content'}), 400

        safe_print(f"\n[ONTRAPORT] Pushing newsletter to Ontraport...")

        ontraport_app_id = os.getenv('ONTRAPORT_APP_ID', '').strip()
        ontraport_api_key = os.getenv('ONTRAPORT_API_KEY', '').strip()

        if not ontraport_app_id or not ontraport_api_key:
            return jsonify({
                'success': False,
                'error': 'Ontraport credentials not configured.'
            }), 400

        ontraport_url = 'https://api.ontraport.com/1/message'

        headers = {
            'Api-Appid': ontraport_app_id,
            'Api-Key': ontraport_api_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        # Generate plain text version
        plain_text = html_to_plain_text(html_content)

        payload = {
            'objectID': ONTRAPORT_CONFIG['object_id'],
            'name': f'Stay In The Loupe {month.title()}',
            'subject': subject,
            'type': 'e-mail',
            'transactional_email': '0',
            'object_type_id': ONTRAPORT_CONFIG['object_type_id'],
            'from': 'custom',
            'send_out_name': ONTRAPORT_CONFIG['from_name'],
            'reply_to_email': ONTRAPORT_CONFIG['reply_to_email'],
            'send_from': ONTRAPORT_CONFIG['from_email'],
            'send_to': 'email',
            'message_body': html_content,
            'text_body': plain_text
        }

        response = req.post(
            ontraport_url,
            headers=headers,
            data=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            safe_print(f"[ONTRAPORT] Success!")
            return jsonify({
                'success': True,
                'message': 'Newsletter pushed to Ontraport',
                'data': result
            })
        else:
            error_msg = f"Ontraport API error: {response.status_code} - {response.text}"
            safe_print(f"[ONTRAPORT ERROR] {error_msg}")
            return jsonify({'success': False, 'error': error_msg}), 500

    except Exception as e:
        safe_print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def html_to_plain_text(html_content):
    """Convert HTML to plain text for email"""
    if not html_content:
        return ''
    text = str(html_content)
    text = re.sub(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>([^<]*)</a>', r'\2 (\1)', text)
    text = re.sub(r'<li[^>]*>', '• ', text)
    text = re.sub(r'</li>', '\n', text)
    text = re.sub(r'<p[^>]*>', '', text)
    text = re.sub(r'</p>', '\n\n', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&nbsp;', ' ')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================================================
# ROUTES - EMAIL TEMPLATE RENDERING
# ============================================================================

@app.route('/api/render-email-template', methods=['POST'])
def render_email_template():
    """Render the email template with newsletter content"""
    try:
        data = request.json
        content = data.get('content', {})
        month = data.get('month', datetime.now().strftime('%B'))
        year = data.get('year', datetime.now().year)
        preheader = data.get('preheader', f"Your monthly jewelry industry insights - {month} {year}")
        images = data.get('images', {})

        safe_print(f"\n[API] Rendering email template for {month} {year}...")

        # Load template
        template_path = os.path.join('templates', 'stay-in-the-loupe-email.html')
        if not os.path.exists(template_path):
            return jsonify({'success': False, 'error': 'Email template not found'}), 404

        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        # Replace basic placeholders
        html = template.replace('{{MONTH}}', month)
        html = html.replace('{{YEAR}}', str(year))
        html = html.replace('{{PREHEADER}}', preheader)

        # Intro content
        html = html.replace('{{INTRO_CONTENT}}', content.get('intro', 'Welcome to this month\'s edition!'))

        # The Brite Spot
        html = html.replace('{{BRITE_SPOT_CONTENT}}', content.get('brite_spot', ''))
        html = html.replace('{{BRITE_SPOT_IMAGE}}', images.get('brite_spot', {}).get('url', 'https://placehold.co/180x180/008181/white?text=Brite+Spot'))

        # Generate Brite Spot bullets if provided
        brite_spot_bullets = ''
        if content.get('brite_spot_bullets'):
            brite_spot_bullets = '<table border="0" cellpadding="0" cellspacing="0" role="presentation" width="100%"><tbody>'
            for bullet in content['brite_spot_bullets']:
                brite_spot_bullets += f'''
                <tr>
                    <td style="padding-top: 6px;" valign="top" width="20">
                        <table border="0" cellpadding="0" cellspacing="0" role="presentation">
                            <tbody><tr><td style="width: 10px; height: 10px; background-color: #31D7CA; border-radius: 50%; font-size: 1px; line-height: 1px;">&nbsp;</td></tr></tbody>
                        </table>
                    </td>
                    <td style="padding-bottom: 10px; padding-left: 10px;" valign="top">
                        <p class="dark-text-secondary" style="margin: 0; font-family: Montserrat, Helvetica, Arial, sans-serif; font-size: 15px; line-height: 24px; font-weight: 400; color: #3B3B3B;">{bullet}</p>
                    </td>
                </tr>'''
            brite_spot_bullets += '</tbody></table>'
        html = html.replace('{{BRITE_SPOT_BULLETS}}', brite_spot_bullets)

        # The Good, The Bad, The Ugly
        the_good = content.get('the_good', {})
        html = html.replace('{{THE_GOOD_SUBTITLE}}', the_good.get('subtitle', 'Good News'))
        html = html.replace('{{THE_GOOD_COPY}}', the_good.get('copy', ''))
        html = html.replace('{{THE_GOOD_URL}}', the_good.get('url', '#'))

        the_bad = content.get('the_bad', {})
        html = html.replace('{{THE_BAD_SUBTITLE}}', the_bad.get('subtitle', 'Cautionary Tale'))
        html = html.replace('{{THE_BAD_COPY}}', the_bad.get('copy', ''))
        html = html.replace('{{THE_BAD_URL}}', the_bad.get('url', '#'))

        the_ugly = content.get('the_ugly', {})
        html = html.replace('{{THE_UGLY_SUBTITLE}}', the_ugly.get('subtitle', 'Unusual Story'))
        html = html.replace('{{THE_UGLY_COPY}}', the_ugly.get('copy', ''))
        html = html.replace('{{THE_UGLY_URL}}', the_ugly.get('url', '#'))

        # Industry News bullets
        industry_news = content.get('industry_news', {})
        bullets = industry_news.get('bullets', [])
        news_bullets_html = ''
        for i, bullet in enumerate(bullets[:5]):
            text = bullet.get('text', bullet) if isinstance(bullet, dict) else str(bullet)
            url = bullet.get('url', '#') if isinstance(bullet, dict) else '#'
            padding_bottom = '16px' if i < len(bullets) - 1 else '0'
            news_bullets_html += f'''
            <tr>
                <td style="padding-right: 12px; padding-top: 2px;" valign="top" width="32">
                    <img alt="" src="https://brite.co/wp-content/uploads/2025/09/tickk.png" style="display: block; width: 24px; height: auto;" width="24" />
                </td>
                <td style="padding-bottom: {padding_bottom};" valign="top">
                    <p class="dark-text-secondary" style="margin: 0; font-family: Montserrat, Helvetica, Arial, sans-serif; font-size: 15px; line-height: 24px; font-weight: 400; color: #3B3B3B;">{text} <a href="{url}" style="color: #008181; text-decoration: underline;">Read more</a></p>
                </td>
            </tr>'''
        html = html.replace('{{INDUSTRY_NEWS_BULLETS}}', news_bullets_html)

        # Industry Pulse
        pulse = content.get('industry_pulse', {})
        html = html.replace('{{INDUSTRY_PULSE_TITLE}}', pulse.get('title', 'Industry Trends'))
        html = html.replace('{{INDUSTRY_PULSE_INTRO}}', pulse.get('intro', ''))
        html = html.replace('{{INDUSTRY_PULSE_H3_1_TITLE}}', pulse.get('h3_1_title', ''))
        html = html.replace('{{INDUSTRY_PULSE_H3_1_CONTENT}}', pulse.get('h3_1_content', ''))
        html = html.replace('{{INDUSTRY_PULSE_H3_2_TITLE}}', pulse.get('h3_2_title', ''))
        html = html.replace('{{INDUSTRY_PULSE_H3_2_CONTENT}}', pulse.get('h3_2_content', ''))
        html = html.replace('{{INDUSTRY_PULSE_IMAGE}}', images.get('industry_pulse', {}).get('url', 'https://placehold.co/180x180/282D3E/white?text=Pulse'))

        # Partner Advantage
        pa = content.get('partner_advantage', {})
        html = html.replace('{{PARTNER_ADVANTAGE_SUBHEADER}}', pa.get('subheader', 'Tips for Success'))
        html = html.replace('{{PARTNER_ADVANTAGE_INTRO}}', pa.get('intro', ''))
        html = html.replace('{{PARTNER_ADVANTAGE_IMAGE}}', images.get('partner_advantage', {}).get('url', 'https://placehold.co/180x180/31D7CA/white?text=Tips'))

        # Partner Advantage tips
        tips = pa.get('tips', [])
        tips_html = ''
        for i, tip in enumerate(tips[:5]):
            title = tip.get('title', '') if isinstance(tip, dict) else ''
            tip_content = tip.get('content', str(tip)) if isinstance(tip, dict) else str(tip)
            padding_bottom = '18px' if i < len(tips) - 1 else '0'
            tips_html += f'''
            <tr>
                <td style="padding-right: 12px; padding-top: 2px;" valign="top" width="32">
                    <img alt="" src="https://brite.co/wp-content/uploads/2025/09/tickk.png" style="display: block; width: 24px; height: auto;" width="24" />
                </td>
                <td style="padding-bottom: {padding_bottom};" valign="top">
                    <p class="dark-text" style="margin: 0 0 4px 0; font-family: Montserrat, Helvetica, Arial, sans-serif; font-size: 16px; line-height: 22px; font-weight: 600; color: #008181;">{title}</p>
                    <p class="dark-text-secondary" style="margin: 0; font-family: Montserrat, Helvetica, Arial, sans-serif; font-size: 15px; line-height: 24px; font-weight: 400; color: #3B3B3B;">{tip_content}</p>
                </td>
            </tr>'''
        html = html.replace('{{PARTNER_ADVANTAGE_TIPS}}', tips_html)

        return jsonify({
            'success': True,
            'html': html,
            'month': month,
            'year': year
        })

    except Exception as e:
        safe_print(f"[API ERROR] Render template: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTES - ARCHIVES
# ============================================================================

@app.route('/api/get-newsletter-archive', methods=['GET'])
def get_newsletter_archive():
    """Get archived newsletters"""
    try:
        archive_file = os.path.join('data', 'archives', 'loupe-archives.json')

        if not os.path.exists(archive_file):
            return jsonify({'success': True, 'newsletters': []})

        with open(archive_file, 'r', encoding='utf-8') as f:
            all_newsletters = json.load(f)

        return jsonify({
            'success': True,
            'newsletters': all_newsletters[:6]
        })

    except Exception as e:
        safe_print(f"[API ERROR] {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/save-newsletter-archive', methods=['POST'])
def save_newsletter_archive():
    """Save newsletter to archive"""
    try:
        data = request.json

        archive_file = os.path.join('data', 'archives', 'loupe-archives.json')

        if os.path.exists(archive_file):
            with open(archive_file, 'r', encoding='utf-8') as f:
                archives = json.load(f)
        else:
            archives = []

        new_entry = {
            'id': f"loupe-{data.get('year')}-{data.get('month')}",
            'month': data.get('month'),
            'year': data.get('year'),
            'date_published': datetime.now().strftime('%Y-%m-%d'),
            'sections': data.get('sections', {}),
            'html_content': data.get('html_content', ''),
            'ontraport_campaign_id': data.get('ontraport_campaign_id')
        }

        archives.insert(0, new_entry)
        archives = archives[:12]

        os.makedirs(os.path.dirname(archive_file), exist_ok=True)
        with open(archive_file, 'w', encoding='utf-8') as f:
            json.dump(archives, f, indent=2)

        return jsonify({'success': True, 'message': 'Newsletter archived'})

    except Exception as e:
        safe_print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTES - DRAFTS (Google Cloud Storage)
# ============================================================================

@app.route('/api/save-draft', methods=['POST'])
def save_draft():
    """Auto-save newsletter draft to GCS"""
    if not gcs_client:
        return jsonify({'success': False, 'error': 'GCS not available'}), 503
    try:
        data = request.json
        month = data.get('month', 'unknown').lower()
        year = data.get('year', datetime.now().year)
        saved_by = data.get('savedBy', 'unknown').split('@')[0].replace('.', '-')
        blob_name = f"drafts/{month}-{year}-{saved_by}.json"

        draft = {
            'month': month,
            'year': year,
            'currentStep': data.get('currentStep'),
            'selectedArticles': data.get('selectedArticles'),
            'generatedContent': data.get('generatedContent'),
            'generatedImages': data.get('generatedImages'),
            'generatedPrompts': data.get('generatedPrompts'),
            'specialSection': data.get('specialSection'),
            'introText': data.get('introText'),
            'briteSpotTitle': data.get('briteSpotTitle'),
            'briteSpotBody': data.get('briteSpotBody'),
            'subjectLine': data.get('subjectLine'),
            'preheader': data.get('preheader'),
            'customLinks': data.get('customLinks'),
            'skippedSections': data.get('skippedSections'),
            'lastSavedBy': data.get('savedBy', 'unknown'),
            'lastSavedAt': datetime.now(CHICAGO_TZ).isoformat()
        }

        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(draft), content_type='application/json')
        return jsonify({'success': True, 'file': blob_name})

    except Exception as e:
        safe_print(f"[DRAFT SAVE ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/list-drafts', methods=['GET'])
def list_drafts():
    """List all saved drafts from GCS"""
    if not gcs_client:
        return jsonify({'success': True, 'drafts': []})
    try:
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blobs = bucket.list_blobs(prefix='drafts/')
        drafts = []
        for blob in blobs:
            if not blob.name.endswith('.json'):
                continue
            data = json.loads(blob.download_as_text())
            drafts.append({
                'month': data.get('month'),
                'year': data.get('year'),
                'currentStep': data.get('currentStep'),
                'lastSavedBy': data.get('lastSavedBy'),
                'lastSavedAt': data.get('lastSavedAt'),
                'filename': blob.name
            })
        # Sort by lastSavedAt descending
        drafts.sort(key=lambda d: d.get('lastSavedAt', ''), reverse=True)
        return jsonify({'success': True, 'drafts': drafts})

    except Exception as e:
        safe_print(f"[DRAFT LIST ERROR] {str(e)}")
        return jsonify({'success': True, 'drafts': []})


@app.route('/api/load-draft', methods=['GET'])
def load_draft():
    """Load a specific draft from GCS"""
    if not gcs_client:
        return jsonify({'success': False, 'error': 'GCS not available'}), 503
    try:
        filename = request.args.get('file')
        if not filename:
            return jsonify({'success': False, 'error': 'No file specified'}), 400

        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(filename)
        data = json.loads(blob.download_as_text())
        return jsonify({'success': True, 'draft': data})

    except Exception as e:
        safe_print(f"[DRAFT LOAD ERROR] {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/publish-draft', methods=['POST'])
def publish_draft():
    """Move a draft from drafts/ to published/ in GCS"""
    if not gcs_client:
        return jsonify({'success': False, 'error': 'GCS not available'}), 503
    try:
        filename = request.json.get('file')
        if not filename:
            return jsonify({'success': False, 'error': 'No file specified'}), 400
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        source_blob = bucket.blob(filename)
        if not source_blob.exists():
            return jsonify({'success': False, 'error': 'Draft not found'}), 404
        published_name = filename.replace('drafts/', 'published/', 1)
        bucket.copy_blob(source_blob, bucket, published_name)
        source_blob.delete()
        safe_print(f"[DRAFT] Published {filename} -> {published_name}")
        return jsonify({'success': True, 'file': published_name})
    except Exception as e:
        safe_print(f"[DRAFT PUBLISH ERROR] {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/list-published', methods=['GET'])
def list_published():
    """List all published newsletters from GCS"""
    if not gcs_client:
        return jsonify({'success': True, 'newsletters': []})
    try:
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix='published/'))
        newsletters = []
        for blob in blobs:
            if blob.name.endswith('.json'):
                data = json.loads(blob.download_as_text())
                gc = data.get('generatedContent', {})
                newsletters.append({
                    'filename': blob.name,
                    'month': data.get('month'),
                    'year': data.get('year'),
                    'lastSavedBy': data.get('lastSavedBy'),
                    'lastSavedAt': data.get('lastSavedAt'),
                    'sections': {
                        'gbu': {'title': gc.get('introduction', 'Newsletter')[:80] if gc.get('introduction') else 'The Good, Bad & Ugly'},
                        'pulse': {'title': gc.get('industry_pulse', {}).get('title', 'Industry Pulse')[:80] if isinstance(gc.get('industry_pulse'), dict) else 'Industry Pulse'},
                        'partner': {'title': gc.get('partner_advantage', {}).get('subheader', 'Partner Advantage')[:80] if isinstance(gc.get('partner_advantage'), dict) else 'Partner Advantage'},
                    }
                })
        newsletters.sort(key=lambda d: d.get('lastSavedAt', ''), reverse=True)
        return jsonify({'success': True, 'newsletters': newsletters})
    except Exception as e:
        safe_print(f"[PUBLISHED LIST ERROR] {str(e)}")
        return jsonify({'success': True, 'newsletters': []})


@app.route('/api/load-published', methods=['GET'])
def load_published():
    """Load a specific published newsletter from GCS"""
    if not gcs_client:
        return jsonify({'success': False, 'error': 'GCS not available'}), 503
    try:
        filename = request.args.get('file')
        if not filename:
            return jsonify({'success': False, 'error': 'No file specified'}), 400
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(filename)
        if not blob.exists():
            return jsonify({'success': False, 'error': 'Not found'}), 404
        data = json.loads(blob.download_as_text())
        return jsonify({'success': True, 'draft': data})
    except Exception as e:
        safe_print(f"[PUBLISHED LOAD ERROR] {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/delete-draft', methods=['DELETE'])
def delete_draft():
    """Delete a draft from GCS (called after Ontraport push)"""
    if not gcs_client:
        return jsonify({'success': True})
    try:
        filename = request.json.get('file')
        if not filename:
            return jsonify({'success': False, 'error': 'No file specified'}), 400

        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(filename)
        if blob.exists():
            blob.delete()
        return jsonify({'success': True})

    except Exception as e:
        safe_print(f"[DRAFT DELETE ERROR] {str(e)}")
        return jsonify({'success': True})  # Non-critical, don't fail


@app.route('/api/delete-published', methods=['DELETE'])
def delete_published():
    """Delete a published newsletter from GCS"""
    if not gcs_client:
        return jsonify({'success': True})
    try:
        filename = request.json.get('file')
        if not filename:
            return jsonify({'success': False, 'error': 'No file specified'}), 400
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(filename)
        if blob.exists():
            blob.delete()
            safe_print(f"[PUBLISHED] Deleted {filename}")
        return jsonify({'success': True})
    except Exception as e:
        safe_print(f"[PUBLISHED DELETE ERROR] {str(e)}")
        return jsonify({'success': True})


# ============================================================================
# ROUTES - SAVED ARTICLES (Google Cloud Storage)
# ============================================================================

SAVED_ARTICLES_BLOB = 'saved-articles/global.json'

@app.route('/api/saved-articles', methods=['GET'])
def get_saved_articles():
    """Get all saved articles from GCS"""
    if not gcs_client:
        return jsonify({'success': True, 'articles': []})
    try:
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(SAVED_ARTICLES_BLOB)
        if blob.exists():
            data = json.loads(blob.download_as_text())
            return jsonify({'success': True, 'articles': data.get('articles', [])})
        return jsonify({'success': True, 'articles': []})
    except Exception as e:
        safe_print(f"[SAVED ARTICLES] Error loading: {str(e)}")
        return jsonify({'success': True, 'articles': []})


@app.route('/api/saved-articles', methods=['POST'])
def add_saved_article():
    """Add an article to the saved articles list"""
    if not gcs_client:
        return jsonify({'success': False, 'error': 'GCS not available'}), 503
    try:
        article = request.json.get('article')
        if not article or not article.get('url'):
            return jsonify({'success': False, 'error': 'Article with URL required'}), 400

        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(SAVED_ARTICLES_BLOB)

        # Load existing
        articles = []
        if blob.exists():
            data = json.loads(blob.download_as_text())
            articles = data.get('articles', [])

        # Check for duplicate URL
        if any(a.get('url') == article['url'] for a in articles):
            return jsonify({'success': True, 'message': 'Already saved', 'articles': articles})

        # Add with timestamp (Chicago time)
        article['dateSaved'] = datetime.now(CHICAGO_TZ).isoformat()
        articles.insert(0, article)

        blob.upload_from_string(json.dumps({'articles': articles}), content_type='application/json')
        return jsonify({'success': True, 'articles': articles})

    except Exception as e:
        safe_print(f"[SAVED ARTICLES] Error saving: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/saved-articles', methods=['DELETE'])
def delete_saved_article():
    """Remove an article from saved articles by URL"""
    if not gcs_client:
        return jsonify({'success': False, 'error': 'GCS not available'}), 503
    try:
        url = request.json.get('url')
        if not url:
            return jsonify({'success': False, 'error': 'URL required'}), 400

        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(SAVED_ARTICLES_BLOB)

        articles = []
        if blob.exists():
            data = json.loads(blob.download_as_text())
            articles = data.get('articles', [])

        articles = [a for a in articles if a.get('url') != url]
        blob.upload_from_string(json.dumps({'articles': articles}), content_type='application/json')
        return jsonify({'success': True, 'articles': articles})

    except Exception as e:
        safe_print(f"[SAVED ARTICLES] Error deleting: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# NOT-GOOD-FIT TRACKING
# ============================================================================

NOT_GOOD_FIT_BLOB = 'feedback/not-good-fit.json'

@app.route('/api/not-good-fit', methods=['POST'])
def track_not_good_fit():
    """Track articles marked as 'not a good fit' for learning"""
    if not gcs_client:
        return jsonify({'success': False, 'error': 'GCS not available'}), 503
    try:
        article = request.json.get('article', {})
        if not article.get('url'):
            return jsonify({'success': False, 'error': 'Article URL required'}), 400

        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(NOT_GOOD_FIT_BLOB)

        entries = []
        if blob.exists():
            data = json.loads(blob.download_as_text())
            entries = data.get('entries', [])

        # Avoid duplicates
        if not any(e.get('url') == article['url'] for e in entries):
            entries.append({
                'url': article['url'],
                'title': article.get('title', ''),
                'publisher': article.get('publisher', ''),
                'markedAt': datetime.now(CHICAGO_TZ).isoformat()
            })
            blob.upload_from_string(json.dumps({'entries': entries}), content_type='application/json')

        return jsonify({'success': True, 'count': len(entries)})

    except Exception as e:
        safe_print(f"[NOT-GOOD-FIT] Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# IMAGE HOSTING - Upload images to GCS for email
# ============================================================================

@app.route('/api/upload-images-to-gcs', methods=['POST'])
def upload_images_to_gcs():
    """Upload newsletter images to GCS and return public URLs"""
    if not gcs_client:
        return jsonify({'success': False, 'error': 'GCS not configured'}), 500

    try:
        data = request.json
        images = data.get('images', {})  # { section_name: base64_data_url }
        month = data.get('month', 'unknown')
        year = data.get('year', datetime.now().year)

        if not images:
            return jsonify({'success': False, 'error': 'No images provided'}), 400

        bucket = gcs_client.bucket(GCS_IMAGES_BUCKET)
        uploaded_urls = {}
        timestamp = datetime.now(CHICAGO_TZ).strftime('%Y%m%d-%H%M%S')

        for section, data_url in images.items():
            if not data_url or not data_url.startswith('data:image'):
                continue

            # Parse base64 data URL
            # Format: data:image/png;base64,iVBORw0KGgo...
            try:
                header, b64_data = data_url.split(',', 1)
                # Extract image format
                img_format = 'png'
                if 'jpeg' in header or 'jpg' in header:
                    img_format = 'jpg'
                elif 'webp' in header:
                    img_format = 'webp'

                # Decode base64
                image_bytes = base64.b64decode(b64_data)

                # Create unique filename
                safe_section = section.replace('_', '-')
                filename = f"newsletters/{year}/{month.lower()}/{timestamp}-{safe_section}.{img_format}"

                # Upload to GCS
                blob = bucket.blob(filename)
                blob.upload_from_string(
                    image_bytes,
                    content_type=f'image/{img_format}'
                )

                # Make publicly accessible
                blob.make_public()

                # Get public URL
                uploaded_urls[section] = blob.public_url
                safe_print(f"[GCS] Uploaded {section} -> {blob.public_url}")

            except Exception as img_error:
                safe_print(f"[GCS] Error uploading {section}: {str(img_error)}")
                continue

        return jsonify({
            'success': True,
            'urls': uploaded_urls,
            'count': len(uploaded_urls)
        })

    except Exception as e:
        safe_print(f"[GCS UPLOAD] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ARTICLE SELECTION TRACKING
# ============================================================================

@app.route('/api/track-selection', methods=['POST'])
def track_selection():
    """Track article selection for learning/analysis"""
    if not gcs_client:
        return jsonify({'success': False, 'error': 'GCS not configured'}), 500

    try:
        data = request.json
        selection = {
            'timestamp': datetime.now(CHICAGO_TZ).isoformat(),
            'app': 'jeweler',
            'section': data.get('section'),  # e.g., 'the_good', 'industry_pulse'
            'article': {
                'url': data.get('url'),
                'title': data.get('title'),
                'headline': data.get('headline'),
                'publisher': data.get('publisher'),
                'snippet': data.get('snippet'),
                'impact': data.get('impact')
            },
            'searchQuery': data.get('searchQuery'),
            'searchSource': data.get('searchSource'),  # 'perplexity' or 'google'
            'timeFilter': data.get('timeFilter'),
            'user': data.get('user', 'unknown'),
            'month': data.get('month'),
            'deselected': data.get('deselected', False)  # True if user unselected
        }

        # Store in GCS: selection-history/jeweler/2026-01.jsonl
        year_month = datetime.now(CHICAGO_TZ).strftime('%Y-%m')
        blob_name = f'selection-history/jeweler/{year_month}.jsonl'

        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)

        # Append to existing file (JSONL format - one JSON per line)
        existing = ''
        if blob.exists():
            existing = blob.download_as_text()

        new_content = existing + json.dumps(selection) + '\n'
        blob.upload_from_string(new_content, content_type='application/jsonl')

        return jsonify({'success': True})
    except Exception as e:
        # Silent fail - don't interrupt user workflow
        safe_print(f"[TRACK] Error: {str(e)}")
        return jsonify({'success': True})  # Return success anyway


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    print(f"\n{'='*60}")
    print(f"  Stay In The Loupe - Jeweler Newsletter Generator")
    print(f"  Running on http://localhost:{port}")
    print(f"{'='*60}\n")

    app.run(host='0.0.0.0', port=port, debug=debug)
