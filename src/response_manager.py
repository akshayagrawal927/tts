import random

class WittyResponseManager:
    """Manages intelligent, non-repetitive witty responses for Pulse AI"""
    
    def __init__(self):
        self.used_responses = {
            'welcome': set(),
            'success': set(),
            'empty': set(),
            'error': set(),
            'processing': set()
        }
        self.response_count = 0
        
        # Welcome messages pool
        self.welcome_messages = [
            "Welcome aboard Pulse AI! I'm Pulse AI—your caffeine-free but hyper-curious analytics sidekick. Ask me anything about your data and I'll happily dig through your data for nuggets of truth, whip up charts, and even spot time-based trends before your coffee cools.",
            
            "Hey there! I'm Pulse AI, your friendly neighborhood data detective. Think of me as Sherlock Holmes but for spreadsheets—minus the pipe, plus the SQL queries. Ready to solve some data mysteries together?",
            
            "Welcome to the data playground! I'm Pulse AI, and I speak fluent SQL, chart-ese, and insight-ish. Point me toward your questions and I'll turn your data into a treasure chest of answers.",
            
            "Greetings, data explorer! Pulse AI here—part analyst, part magician, all algorithms. I transform your curiosity into charts, your questions into insights, and your confusion into clarity. What shall we discover first?",

            "Hello Leader! You're now plugged into Pulse AI—where questions meet instant intelligence. Consider me your strategy co-pilot."
        ]
        
        # Success response templates based on result count
        self.success_responses = {
            'single': [
                "Bullseye! I tracked down exactly **1 record** that fits like a glove.",
                "Bullseye—1 record found, sharper than a boardroom briefing.",
                "Bingo! Found your needle in the haystack—exactly **1 record** that matches perfectly.",
                "Perfect shot! Landed on exactly **1 record** that's precisely what you're after.",
                "Gold! Struck exactly **1 record** that's right on target."
            ],
            'small': [  # 2-10 records
                "Got it—**{count} records** neatly fetched and ready for your inspection. Small but mighty.",
                "Just the right slice—{count} results ready for a quick leadership scan."
                "{count} records lined up—focused insights without the clutter."
                "Nice! Pulled up **{count} records** that fit the bill. Quality over quantity, right?",
                "Sweet spot! Found **{count} records** that match your criteria—just enough to be interesting.",
                "Perfect handful! **{count} records** served fresh from the database buffet.",
            ],
            'medium': [  # 11-100 records
                " {count} records retrieved—enough to map trends, lean enough to stay sharp.",
                "A healthy dataset—{count} entries for you to read between the lines.",
                "Nice catch! I reeled in **{count} records** matching your specs—plenty to analyze, not enough to drown in.",
                "Excellent haul! **{count} records** ready for action—enough data to tell a story without writing a novel.",
                "Jackpot! **{count} records** that hit the sweet spot between 'too little' and 'too much.'",
                "Beautiful! **{count} records** lined up like data soldiers, ready for your command."
            ],
            'large': [  # 100+ records
                "Big wave incoming—{count} records. Let's cut through and surface the strategic signals.",
                "Data firehose detected—{count} rows captured. Time to filter for what really matters."
                "Data avalanche alert! I've hauled in **{count} records**. Bring your biggest spreadsheet appetite.",
                "Wow! **{count} records** incoming—hope you've got your data processing pants on!",
                "Holy datasets, Batman! **{count} records** at your service. Time to put on your analyst cape!",
                "Monster haul! **{count} records** ready to party—this is where the real fun begins."
            ]
        }
        
        # Empty result responses
        self.empty_responses = {
            'date_based': [
                "No activity in that time frame—either calm seas, or filters too tight. Expand range?",

                "Hmm, my time machine came back empty. No data for that period. Maybe nothing happened, or maybe the range is playing hard to get.\nWant me to widen the time window or try a different angle?",
                
                "Crickets... Nothing showed up for that timeframe. Either it was a really quiet period, or the data is playing hide-and-seek.\nShall we try casting a wider net?",
                
                "Time travel complete, but I came back empty-handed. That period seems to be a data desert.\nWant to explore a different time zone or adjust the search?",
                
                "Plot twist: the data for that period is apparently on vacation. No records found.\nLet's try a broader date range or switch up the approach?"
            ],
            'general': [
                "Searched the landscape—no matches. Could be filters playing defense. Try widening?",

                "I scoured every corner and... nada. Could be a spelling twist, a filter too tight, or that record just doesn't exist.\nShall we loosen the filters or brainstorm a new lead?",
                
                "Well, this is awkward. My database dive came up dry. Maybe we're looking for a unicorn?\nLet's try rephrasing or relaxing those search criteria.",
                
                "Houston, we have a... nothing. Zero matches for that query. The data might be hiding under a different name.\nWant to try a different search angle?",
                
                "Mission: Find data. Status: Mission impossible. Nothing matches those criteria right now.\nShall we adjust our search strategy or try a different approach?"
            ]
        }
        
        # Error response templates
        self.error_responses = {
            'invalid_query': [
                "I speak analytics, not riddles. Frame it in data terms, and I'm all yours.",

                "Appreciate the enthusiasm, but I'm a data devotee, not a life coach.\nAsk me about sales, customers, product metrics, or time-based trends and I'm all ears (well...processors).",
                
                "I love the creativity, but I'm more 'SQL wizard' than 'general knowledge guru.'\nTry me with some juicy data questions—sales figures, customer insights, that sort of thing!",
                
                "That's outside my wheelhouse! I'm like a specialized chef—amazing with data dishes, hopeless with anything else.\nWhat data mysteries can I solve for you today?",
                
                "Nice try, but I'm a one-trick pony—and that trick is turning data into insights!\nGot any burning questions about your business data?"
            ],
            'table_error': [
                "Access denied—that table’s behind a locked door. Check permissions or ask support",

                "Knock knock—no answer. Looks like the table structure may have changed.\nDouble-check those table names. Current VIP list: sales_data, customer_data, sapproduct.",
                
                "Table not found! It's like showing up to a party that moved venues.\nOur current guest list includes: sales_data, customer_data, sapproduct.",
                
                "Oops! That table seems to have gone on a coffee break.\nTry these available options: sales_data, customer_data, sapproduct.",
                
                "Table troubles! It's either incognito or doesn't exist.\nStick with the classics: sales_data, customer_data, sapproduct."
            ],
            'column_error': [
                "That column's off the map—possibly renamed or retired. Want me to show what's on the field?",

                "Column? Never heard of it. Maybe it's under a different alias.\nTry a new name or ask what fields I *do* know about.",
                
                "That column is playing hide-and-seek and winning!\nMight be going by a different name—want to see what's actually available?",
                
                "Column not found! It's either in witness protection or using a fake ID.\nLet's explore what fields are actually in the table.",
                
                "Mystery column alert! Either it doesn't exist or it's using an alias.\nShall we investigate what's really available in there?"
            ],
            'syntax_error': [

                "Query tripped over syntax. Let's simplify—one metric, one filter, one goal.",

                "My SQL parser just raised an eyebrow. Let's simplify: ask one thing at a time with clear names and we'll be best friends again.",
                
                "Syntax hiccup! My brain got a little tangled there.\nLet's break it down—one question at a time works best for me.",
                
                "Whoops! That query made my circuits do a little dance.\nSimpler questions help me give you better answers—let's try again!",
                
                "SQL syntax says 'nope!' Let's untangle this together.\nOne clear question at a time is my sweet spot."
            ],
            'data_type_error': [
                "Type mismatch detected—recast and I'll process seamlessly.",
                
                "Oil and water don't mix—and neither do those data types.\nCould you rephrase with a bit more precision so I can blend it cleanly?",
                
                "Data type clash! Like trying to add apples and oranges.\nLet's adjust the query so everything plays nicely together.",
                
                "Type mismatch alert! My data mixer is having compatibility issues.\nA little rephrasing should smooth things out.",
                
                "Data types are having a disagreement. Think oil and water, but geekier.\nLet's rephrase to make everyone happy."
            ],
            'general_error': [
                "Backend hiccup—server or schema turbulence. Retry or let's adjust.",

                "Something hiccupped on the database side. Could be a passing cloud in the server sky.\nTry again with a tighter query or ping support if the gremlins persist.",
                
                "Database burp! Sometimes the servers need a moment to collect themselves.\nGive it another shot, or contact support if it keeps being moody.",
                
                "Technical timeout! Even databases need coffee breaks sometimes.\nTry once more, or reach out to support if this becomes a habit.",
                
                "Server shenanigans detected! The database is having a moment.\nRetry in a few seconds, or escalate to support if it persists."
            ],
            'time_no_data': [
                "No data found for that time period. The filters are correct, but there's simply no activity recorded during that timeframe.",
                "Timeline came up empty—either a quiet period or data hasn't been logged yet for that range.",
                "No records in that date range. Try expanding the time window or checking if data collection was active then."
            ]
        }
        
        # Processing messages
        self.processing_messages = [
            "Running the numbers… insights en route",
            "Query in motion—your data story is loading.",
            "Crunching numbers, wrangling rows... give me a second to work my Pulse AI magic.",
            "Diving deep into the data ocean—surfacing with insights in 3, 2, 1...",
            "SQL spells are brewing! Your data insights are cooking up nicely...",
            "Processing at warp speed—transforming your question into data gold...",
            "Targeting your answer with laser precision—almost got it...",
            "Data rockets launching! Prepare for insight touchdown..."
        ]

        self.greeting_responses = [
            "Hello Leader! You're now plugged into Pulse AI—where questions meet instant intelligence. Consider me your strategy co-pilot."

            "Hi! I'm Pulse AI, your friendly data assistant. I specialize in analyzing your data and turning your questions into insights. What would you like to explore today?",
            
            "Hello there! Pulse AI here - ready to dive into your data and uncover some interesting insights. Ask me anything about your sales, customers, products, or any other data you'd like to analyze!",
            
            "Good to meet you! I'm Pulse AI, your AI-powered analytics companion. I can help you query your database, generate charts, and discover valuable business insights. What data mystery shall we solve together?",
            
            "Hi! I'm Pulse AI - think of me as your personal data detective. I love turning complex queries into simple answers and transforming raw data into actionable insights. How can I help you today?",
            
            "Greetings! Pulse AI at your service. I'm here to make your data sing and your insights shine. Whether you need sales reports, customer analytics, or trend analysis - I've got you covered!",
            
            "Hello! I'm Pulse AI, your caffeinated (well, algorithmically speaking) data companion. Ready to transform your curiosity into clear, actionable insights from your data. What shall we discover first?"
        ]
        
    
    def get_greeting_response(self) -> str:
        """Get a friendly greeting response"""
        import random
        return random.choice(self.greeting_responses)
    
    def get_welcome_message(self) -> str:
        """Get a fresh welcome message"""
        available = [msg for i, msg in enumerate(self.welcome_messages) 
                    if i not in self.used_responses['welcome']]
        
        if not available:
            self.used_responses['welcome'].clear()
            available = self.welcome_messages
        
        selected_idx = self.welcome_messages.index(random.choice(available))
        self.used_responses['welcome'].add(selected_idx)
        return self.welcome_messages[selected_idx]
    
    def get_success_message(self, result_count: int, has_insights: bool = False) -> str:
        """Generate success message based on result count with optional insight flavor"""
        if result_count == 0:
            return self.get_empty_message()
        elif result_count == 1:
            category = 'single'
        elif result_count <= 10:
            category = 'small'
        elif result_count <= 100:
            category = 'medium'
        else:
            category = 'large'
        
        # Get fresh response
        available = [msg for i, msg in enumerate(self.success_responses[category]) 
                    if i not in self.used_responses['success']]
        
        if not available:
            self.used_responses['success'].clear()
            available = self.success_responses[category]
        
        template = random.choice(available)
        selected_idx = self.success_responses[category].index(template)
        self.used_responses['success'].add(selected_idx)
        
        if category != 'single':
            return template.format(count=result_count)
        return template
    
    def get_empty_message(self, is_date_query: bool = False) -> str:
        """Get empty result message"""
        category = 'date_based' if is_date_query else 'general'
        
        available = [msg for i, msg in enumerate(self.empty_responses[category]) 
                    if i not in self.used_responses['empty']]
        
        if not available:
            self.used_responses['empty'].clear()
            available = self.empty_responses[category]
        
        selected = random.choice(available)
        selected_idx = self.empty_responses[category].index(selected)
        self.used_responses['empty'].add(selected_idx)
        return selected
    
    def get_error_message(self, error_type: str) -> str:
        """Get witty error message by type"""
        if error_type not in self.error_responses:
            error_type = 'general_error'
        
        available = [msg for i, msg in enumerate(self.error_responses[error_type]) 
                    if i not in self.used_responses['error']]
        
        if not available:
            self.used_responses['error'].clear()
            available = self.error_responses[error_type]
        
        selected = random.choice(available)
        selected_idx = self.error_responses[error_type].index(selected)
        self.used_responses['error'].add(selected_idx)
        return selected
    
    def get_processing_message(self) -> str:
        """Get a witty processing message"""
        available = [msg for i, msg in enumerate(self.processing_messages) 
                    if i not in self.used_responses['processing']]
        
        if not available:
            self.used_responses['processing'].clear()
            available = self.processing_messages
        
        selected = random.choice(available)
        selected_idx = self.processing_messages.index(selected)
        self.used_responses['processing'].add(selected_idx)
        return selected
    
    def add_personality_touch(self, base_response: str, context: dict) -> str:
        """Add personality touches based on context"""
        self.response_count += 1
        
        # Add occasional personality touches
        if self.response_count % 5 == 0:
            touches = [
                "\n\n*Pulse AI tip: I get more caffeinated with each query—keep 'em coming!*",
                "\n\n*Pro tip: The more specific your questions, the more impressive my answers become!*",
                "\n\n*Pulse AI wisdom: Every great insight starts with a curious question!*"
            ]
            base_response += random.choice(touches)
        
        return base_response
