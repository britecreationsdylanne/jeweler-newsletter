"""
Stay In The Loupe - Jeweler Newsletter Configuration
Brand guidelines and newsletter settings for jeweler partners
"""

# Jewelry news sources for search queries
JEWELRY_NEWS_SOURCES = [
    "nationaljeweler.com",
    "jckonline.com",
    "instoremag.com",
    "southernjewelrynews.com",
    "jewellermagazine.com"
]

# Content filters - what to include/exclude
CONTENT_FILTERS = {
    "include": [
        "jewelry",
        "jeweler",
        "diamonds",
        "gemstones",
        "watches",
        "engagement rings",
        "wedding bands",
        "fine jewelry",
        "jewelry design",
        "jewelry trends",
        "jewelry pricing",
        "jewelry heists",
        "jewelry theft",
        "jewelry insurance",
        "luxury goods",
        "precious metals",
        "gold prices",
        "platinum",
        "lab-grown diamonds"
    ],
    "exclude": [
        # Personnel/people news exclusions (as specified by user)
        "promoted to",
        "announces promotion",
        "new CEO",
        "new president",
        "executive appointment",
        "joins as",
        "named to",
        "leadership change",
        "personnel announcement",
        "new hire",
        "appointed as",
        "steps down",
        "retires from",
        "death",
        "obituary",
        "passed away",
        "in memoriam",
        # Other exclusions
        "political",
        "election"
    ]
}

# Ontraport configuration for jeweler newsletter
ONTRAPORT_CONFIG = {
    "object_id": "7",
    "object_type_id": "0",
    "from_email": "jeweler@brite.co",
    "reply_to_email": "jeweler@brite.co",
    "from_name": "BriteCo Insurance"
}

# Team members for preview emails
TEAM_MEMBERS = [
    {"name": "John Ortbal", "email": "john.ortbal@brite.co"},
    {"name": "Stephanie Lynn", "email": "stephanie.lynn@brite.co"},
    {"name": "Dylanne Crugnale", "email": "dylanne.crugnale@brite.co"},
    {"name": "Rachel Akmakjian", "email": "rachel.akmakjian@brite.co"},
    {"name": "Sam McGregor", "email": "sam.mcregor@brite.co"}
]

# Google Drive folder for document export
GOOGLE_DRIVE_FOLDER_ID = "1aXvnR1nqh_0HBXmqLbYSBGnunOi8npsy"

# Brand voice for AI content generation
BRAND_VOICE = {
    "tone": "Professional but approachable with a touch of wit — clever wordplay and light humor welcome but never forced. Knowledgeable about the jewelry industry, supportive of jeweler partners",
    "style": "Clear, concise, actionable, with personality — a wink, not a wisecrack",
    "perspective": "We help jewelers protect their clients' most precious possessions",
    "wit_guidance": [
        "Light wordplay and puns are encouraged, especially in headlines",
        "Keep humor subtle — a clever turn of phrase, not a joke",
        "Never sacrifice clarity for cleverness",
        "The Good/Bad/Ugly section can be the wittiest; Industry Pulse should stay more analytical"
    ],
    "avoid": [
        "Overly salesy language",
        "Jargon without explanation",
        "Non-jewelry content",
        "Political content",
        "Competitor bashing",
        "Personnel/people news (deaths, promotions, etc.)",
        "Over-the-top humor or forced jokes"
    ]
}

# Newsletter section specifications
SECTION_SPECS = {
    "the_good_bad_ugly": {
        "description": "3 distinct sections with a fun spin on jewelry news",
        "sections": ["the_good", "the_bad", "the_ugly"],
        "each_section": {
            "subtitle_max_words": 8,
            "copy_max_words": 30,
            "copy_sentences": "1-2",
            "requires_image": True,
            "requires_hyperlink": True
        },
        "focus_areas": [
            "jewelry pricing",
            "jewelry design",
            "jewelry heists",
            "jewelry-related stories",
            "positive jewelry news (The Good)",
            "negative jewelry news (The Bad)",
            "bizarre/unusual jewelry stories (The Ugly)"
        ]
    },
    "brite_spot": {
        "description": "Introduction to a new feature, tool, product, or big announcement",
        "subheader_max_words": 15,
        "body_max_words": 100,
        "requires_image": True,
        "no_research_needed": True
    },
    "industry_pulse": {
        "description": "Topical industry news piece(s) relevant to jeweler partners",
        "requires_image": True,
        "image_includes_title": True,
        "structure": {
            "intro": {
                "paragraphs": "1-2",
                "sentences_per_paragraph": "1-4",
                "max_words_per_paragraph": 50
            },
            "h3_sections": {
                "count": 2,
                "h3_max_words": 10,
                "supporting_paragraphs": "1-2",
                "max_words_per_paragraph": 60
            }
        },
        "articles_to_select": "2-5",
        "combines_into_one_story": True
    },
    "partner_advantage": {
        "description": "Tips and insights for jeweler partners",
        "subheader_max_words": 15,
        "requires_image": True,
        "structure": {
            "intro_paragraph": True,
            "bullet_points": 5,
            "bullet_title_max_words": 10,
            "bullet_support_sentences": "1-3"
        },
        "articles_to_select": "1-2"
    },
    "industry_news": {
        "description": "Topical news stories or general jeweler tips with links",
        "bullet_points": 5,
        "characteristics": [
            "catchy",
            "include hyperlinks to source",
            "1 sentence each",
            "treat each as a title"
        ]
    }
}

# Writing style guide for jeweler newsletter
WRITING_STYLE_GUIDE = {
    "introduction": {
        "patterns": [
            "Start with a seasonal/monthly reference",
            "Use 'we' to create partnership feel",
            "Mention what's inside the newsletter",
            "Keep it warm but brief (2-3 sentences max)"
        ],
        "example_openers": [
            "Happy [Month]! Here's what's sparkling in the jewelry world this month.",
            "Welcome to [Month]'s Stay In The Loupe!",
            "As we head into [season], here's the latest from the jewelry industry.",
            "[Month] is here, and we've got the insights to keep you ahead of the curve.",
            "[Month] just arrived — and it brought some gems (pun fully intended)."
        ],
        "phrases_to_use": [
            "keeping you in the loop",
            "staying ahead of trends",
            "industry insights",
            "your success"
        ]
    },
    "brite_spot": {
        "patterns": [
            "Lead with the benefit to jewelers",
            "Mention specific BriteCo features or updates",
            "Include a call-to-action when relevant",
            "Use excitement without being salesy"
        ],
        "max_words": 100
    },
    "the_good_bad_ugly": {
        "patterns": [
            "The Good: Positive jewelry news (new designs, trends, sales records)",
            "The Bad: Cautionary tales (heists, thefts, scams, market downturns)",
            "The Ugly: Bizarre, unusual, or eyebrow-raising jewelry stories",
            "Each section: catchy subtitle (8 words max), brief copy (30 words max)"
        ],
        "tone_notes": [
            "Fun and engaging — this is the wittiest section",
            "Clever headlines with wordplay and light puns",
            "Brief but impactful",
            "Always include source links",
            "A dash of humor in the copy is encouraged"
        ]
    },
    "industry_pulse": {
        "patterns": [
            "Combine 2-5 related articles into one cohesive story",
            "Start with a compelling intro (1-2 paragraphs, 50 words max each)",
            "Include 2 H3 subheadings (10 words max each)",
            "Each H3 has 1-2 supporting paragraphs (60 words max each)",
            "Include hyperlinks as sources throughout"
        ],
        "tone_notes": [
            "Analytical but accessible",
            "Data-driven when possible",
            "Connect to jeweler relevance",
            "Professional industry voice with subtle personality",
            "A light touch of wit in transitions is fine, but keep analysis sharp"
        ]
    },
    "partner_advantage": {
        "patterns": [
            "Brief intro paragraph setting up the tips — inject light personality",
            "5 bullet points with mini-titles (10 words max)",
            "Each bullet has 1-3 supporting sentences",
            "Include hyperlinks for further reading",
            "Mini-titles can use clever phrasing where it fits naturally"
        ],
        "categories": [
            "Sales techniques",
            "Client relationships",
            "Industry best practices",
            "Technology adoption",
            "Market insights"
        ]
    },
    "industry_news": {
        "patterns": [
            "5 bullet points",
            "Each is a single catchy sentence",
            "Include hyperlink to source",
            "Treat as headlines/titles",
            "Mix of topics: trends, market news, technology, etc."
        ],
        "avoid": [
            "Personnel news (deaths, promotions)",
            "Non-jewelry content",
            "Political commentary"
        ]
    }
}

# Brand check rules for AI content review
BRAND_CHECK_RULES = {
    "tone_and_voice": {
        "should_be": "Professional but approachable with subtle wit, knowledgeable, supportive",
        "should_avoid": "Overly formal, salesy, condescending language, or forced/over-the-top humor"
    },
    "word_count_limits": {
        "brite_spot_body": 100,
        "good_bad_ugly_copy": 30,
        "industry_pulse_intro_paragraph": 50,
        "industry_pulse_h3_paragraph": 60,
        "partner_advantage_bullet_title": 10
    },
    "required_elements": {
        "the_good_bad_ugly": ["subtitle", "copy", "image", "hyperlink"],
        "industry_pulse": ["intro", "h3_sections", "hyperlinks"],
        "partner_advantage": ["intro", "bullet_points", "hyperlinks"],
        "industry_news": ["5_bullets", "hyperlinks"]
    },
    "content_filters": {
        "exclude_topics": [
            "personnel announcements",
            "deaths/obituaries",
            "political content",
            "non-jewelry topics"
        ]
    }
}
