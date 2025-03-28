📌 how3 score benchmarks guide for how3.io
1️⃣ User Growth 
(How quickly are real users joining and using this crypto?)
Metrics:
Daily Active Addresses (DAA)
Transaction Volume Growth YoY (%)
Bridge Volume (for L2 & Infrastructure projects)
Calculation Method (by Category):
Lending, DeFi, CeFi, GameFi, DePIN: Daily Active Users & transactions via TheGraph
L1/L2 & Infrastructure: Bridge volume, transaction volume via TheGraph
AI Agents: API call counts & interaction metrics via Dune Analytics
Source: TheGraph & Dune Analytics
2️⃣ Earnings Quality 
(Is this crypto generating stable, sustainable revenues?)
Metrics:
Revenue Stability (Quarter-over-quarter volatility)
Revenue Diversification (Number of revenue sources & largest revenue source %)
Calculation Method (by Category):
Lending, DeFi: Protocol fees collected, interest earnings via TheGraph
CeFi: Transaction fees, subscription revenue via Dune Analytics
GameFi: Revenue from marketplace/NFT sales via TheGraph
DePIN: Node earnings and subscription revenue via TheGraph
AI Agents: Subscription revenue via Dune Analytics
L1/L2 & Infrastructure: Transaction fee revenue via TheGraph
Source: TheGraph & Dune Analytics
3️⃣ Fair Value 
(Is the market price sensible compared to fundamentals?)
Metrics:
Market Cap-to-Revenue Multiple
Adjustments: Token Inflation Rate, Whale Concentration (Gini coefficient)
Calculation Method:
Calculate annualized revenue (via TheGraph & Dune Analytics).
Calculate Market Cap-to-Revenue ratio (Market Cap sourced from CoinGecko API).
Adjust score negatively if token inflation is high (inflation rate via supply tracking).
Adjust negatively if whale concentration is high (wallet analysis via TheGraph).
Applicability: Same method applies across all categories.
Source: TheGraph, Dune Analytics, CoinGecko API
4️⃣ Safety & Stability 
(Can I trust this crypto's governance, security, and long-term design?)
Metrics:
Validator/Node Count & Geographical Decentralization
Governance Participation Rate (voting participation %)
Historical Security Incidents & Audit Quality
Calculation Method (by Category):
L1/L2, Infrastructure, DePIN: Validator/Node count and distribution via TheGraph
All Categories: Governance voting data via TheGraph
Historical Incidents: Manual input (quarterly updates)
Source: TheGraph, manual quarterly security review (feasible with quarterly manual checks)
🎯 Supplementary Metric: Tokenomics Simplified (additional contextual insight)
Metrics:
Token Inflation Rate (% increase in circulating supply)
Token Distribution Concentration (Gini Coefficient)
Presentation: Provide contextual insights under the "Fair Value" benchmark.
Calculation Method:
Inflation rate calculated from circulating vs. total/emission schedules (via TheGraph).
Gini coefficient via wallet distribution analysis (TheGraph wallet data).
Source: TheGraph
🚀 Data Sources Summary (Feasibility):
TheGraph (primary): All on-chain activity, revenue, governance, wallet analysis
Dune Analytics: Alternative/complementary source for deeper SQL-based analytics
CoinGecko API: Reliable source for Market Cap data
Manual Inputs: Only for security audits/historical incidents (done quarterly).
This simplified yet powerful benchmark system ensures retail investors clearly understand each project's fundamentals without crypto-jargon overload, perfectly aligning with your vision for how3.io.

Simplified how3 Score Calculation (0-100)
For each benchmark, convert the raw numbers to percentile rankings against all top 300 cryptos in that category:
Top 10% → Score: 90-100 (Excellent)
10%-30% → Score: 70-89 (Strong)
30%-70% → Score: 40-69 (Average)
70%-90% → Score: 20-39 (Weak)
Bottom 10% → Score: 0-19 (Poor)
Final how3 Composite Score: (User Growth Score + Earnings Quality Score + Fair Value Score + Safety Score) / 4

Backend Developer Tasks (Simplified)
Primary Responsibilities:
Fetch specific raw data from TheGraph subgraphs or Dune Analytics SQL queries.
Ignore irrelevant crypto metrics (like social media followers, trading volume, etc.).
What exactly to fetch per benchmark:
Benchmark
Exactly what to Fetch (JSON format suggested)
User Growth
Daily active addresses, Daily transaction count, Daily bridge volume (if applicable)
Earnings Quality
Daily revenue (fees, subscriptions, node earnings), Monthly historical revenue for stability calculation
Fair Value
Current market cap, Annualized revenue, Token inflation/emission rates, Wallet distribution percentages (top 10 wallet holdings %)
Safety & Stability
Node/Validator count, Governance participation (%), Multisig wallet use (boolean), Regulatory audit passed (boolean), attack cost

Backend Tools & Methods:
GraphQL API calls: Fetch data from TheGraph
Dune Analytics Queries: Write basic SQL to fetch required aggregated data if unavailable on TheGraph.
Data Cleaning & Storage: Normalize & store data clearly in your database for frontend consumption.
Frontend Developer Tasks (Simplified)
Primary Responsibilities:
Receive structured benchmark scores (0-100) and explanatory metrics from backend API.
Display data cleanly according to design wireframe, with interactive UI,UX microinteractions etc
Data Format (JSON expected from backend):

{
  "crypto": "Ethereum",
  "category": "L1/L2",
  "benchmarks": {
	"userGrowth": 85,
	"earningsQuality": 78,
	"fairValue": 61,
	"safetyStability": 92
  },
  "compositeScore": 79,
  "explanation": {
	"userGrowth": "Ethereum has excellent adoption with X daily active users.",
	"earningsQuality": "Stable revenue from transaction fees of $Y daily.",
	"fairValue": "Market cap-to-revenue multiple is Zx, fair compared to industry.",
	"safetyStability": "Strong governance participation and decentralized validator set."
  }
}

Use simple charts (diamond indicators).

Show benchmark score clearly (0-100 scale) and color-code indicators based on scores:
0-39 (red), 40-69 (yellow), 70-100 (green)
Simple, explanatory tooltips to help users understand each score's meaning.

Tokenomics Highlight (Separate Section)
Simplified as a short, separate "Tokenomics Health" highlight box on each crypto page:
Inflation Rate (High/Moderate/Low)
Whale Concentration (Centralized/Fairly Distributed)
Emission/Burn Rate (Sustainable/Unsustainable)
This ensures tokenomics aren't overlooked and simplifies user comprehension without cluttering main benchmarks.

And finally 
📊 "Risk Volatility Chart" 
This metric tells retail investors clearly how much the price of a cryptocurrency might reasonably move up or down within a year based on historical data.
It's essentially a simplified volatility indicator, typically using historical annualized volatility:
🔍 Calculation Method (Standardized & Simple):
Get Daily Closing Prices


Fetch the daily closing price data for the crypto asset for the past year (365 days).
Calculate Daily Returns


For each day, calculate the percentage daily return:
Daily Return = (Today's Closing Price - Yesterday's Closing Price) / Yesterday's Closing Price

Calculate Annualized Volatility
Compute standard deviation (σ) of these daily returns.
Annualize it (crypto markets typically trade ~365 days/year):
Annualized Volatility = Daily Standard Deviation × √365

Presenting the Metric Clearly
Convert to percentage and simplify:
Expected Annual Movement (%) = Annualized Volatility × 100
Show clearly as ± X%:
Example: "±42% expected annual movement."
Simple Example:
If Ethereum has a daily returns standard deviation of 2.2%:
Annualized Volatility = 2.2% × √365 ≈ 42%
Clearly shown as:
"Ethereum's price may move ±42% in a year based on historical volatility."
Use coingecko/coinmarketcap api for this

Final json to share to front end is -

{
  "expectedAnnualMovement": 42,
  "currentPrice": 3000,
  "upsidePrice": 4260,
  "downsidePrice": 1740
}


how3.io: simple crypto analytics
Executive Summary
how3.io aims to revolutionize crypto investing by providing intuitive, jargon-free analytics focused on fundamental value rather than trading indicators. 
Our platform will help users make informed investment decisions by categorizing crypto projects as undervalued, aptly valued, or overvalued based on category-specific metrics and industry benchmarks.
Unlike existing platforms that overwhelm users with complex charts and raw data, how3.io translates crypto analytics into accessible insights, focusing on what the numbers actually mean for investors seeking long-term value.
1. Product Vision
1.1 Problem Statement
The crypto analytics space is dominated by platforms that:
Overwhelm users with complex trading indicators and technical jargon
Provide raw data without meaningful context or interpretation
Use inconsistent methodologies across different projects and categories
Fail to distinguish between speculative hype and fundamental value
Create barriers to entry for non-technical investors
1.2 Solution Overview
how3.io will solve these problems by:
Focusing on fundamental metrics that indicate real value across different crypto categories
Translating complex data into accessible insights with plain language explanations
Providing standardized evaluations across different project types
Creating a visual, interactive experience that feels more like a premium consumer app than a financial tool
Empowering users to make informed investment decisions based on project fundamentals
1.3 Target Audience
Retail investors looking for long-term crypto investments
High Net Worth Individuals (HNIs) seeking efficient ways to evaluate crypto opportunities
Crypto-curious traditional investors transitioning from stock markets
Users who want to cut through the noise and identify value-based investments
Investors who prioritize understanding fundamentals over trading indicators
2. Core Feature Requirements
2.1 Crypto Project Classification System
Requirements:
Categorize the top 300 cryptocurrencies by market cap into distinct business categories
Provide standardized classification data for each project including:
Token Type (utility or governance.) // NR - Memecoins, Security, Stablecoins, Wrapped tokens.
Business Category (specific sector the project operates in) Infra, RWA, AI Agent, Gamefi, Defi, Cefi, Depin
Unique Selling Proposition (differentiator)
Industry Impact (influence on its sector)
Implementation Details:
Develop a classification taxonomy with clear criteria for each category
Create an automated classification system supplemented by manual review
Update classifications quarterly or upon significant project pivots
2.2 Category-Specific Metrics System
Requirements:
Implement tailored primary metrics for evaluating projects based on their category:


Lending Protocols: Total Value Locked (TVL), Loan-to-Value Ratios
L1/L2 Blockchains: Transaction Fees Generated, TPS, Source of Revenue (Optional -
 Active Addresses)
AI Projects: Revenue, Adoption Metrics, Utility Measures
DeFi Protocols: Revenue, Unique Users, Protocol Efficiency measured like this -

Protocol
Revenue/TVL
Liquidity Utilization
Gas per Borrow
APY Stability (σ)
Aave V3
0.015
72%
180k gas
1.8%


Compound
0.012
68%
210k gas
2.3%
MakerDAO
0.008
85%
550k gas
4.1%


RWA (Real World Assets): Asset Backing Ratio, Underlying Asset Performance
We are not focusing on these 2 sectors, but they will have default values ; as scraped using our algorithm
Governance: Voter Participation, Proposal Implementation Rate
Memecoins: Community Growth, Social Engagement
Implementation Details:
Develop automated data collection systems for each metric using APIs
Create normalized scoring systems to enable cross-category comparisons
Provide clear explanations of what each metric means and why it matters
2.3 Valuation Framework
Requirements:
Create a proprietary valuation model that determines if projects are:
Undervalued: Strong fundamentals relative to market cap.
Aptly Valued: Market cap appropriately reflects fundamentals
Overvalued: Market cap exceeds what fundamentals justify
	
	The methodology you use specific to the category of the said coin, 
	You will find a mean average of the (fair pricing model)
	all category based tokens.
Base valuations on industry-specific multiples and benchmarks


Update valuations daily based on market conditions and project metrics
Implementation Details:
Develop benchmark data for each category based on industry leaders
Create a standardized scoring system across four dimensions:
Growth (user adoption, volume growth)
Value (what multiples its trading at, relevant to category)
Profit/Revenue (financial sustainability wrt to their qoq growth)
Health (token economics, treasury management, and overall functioning)
Implement visual indicators that clearly communicate valuation assessments
2.4 User Interface and Experience
Requirements:
Create a clean, intuitive interface similar to www.revvinvest.com but adapted for cryptocurrencies and blockchain projects which are tokenized.
Implement interactive elements that allow users to explore project metrics
Develop comprehensive but accessible project pages with:
Project Overview (what the project does in plain language)
Key Metrics (with explanations of why they matter)
Performance Benchmarks (compared to category peers)
Analysis of Strengths and Weaknesses using GROK
Risk Assessment
Implementation Details:
Develop a responsive web application (I suggest React, Next.Js)
Create interactive data visualizations that contextualize metrics
Implement intuitive navigation between project profiles
For MVP, focus on core functionality with simpler UI components
2.5 Data Collection and Processing System
Requirements:
Establish reliable data feeds from multiple sources including:
Blockchain explorers (Etherscan, Solscan)
Market data providers (CoinGecko, CoinMarketCap)
DeFi analytics platforms (DefiLlama)
Social metrics providers (Grok ; X, Reddit)
Implement data normalization and quality control processes
Create a robust database architecture for efficient queries
Implementation Details:
Develop API integration with primary data sources
Implement data validation and cleaning procedures
Create automated alerts for unusual data patterns
Establish backup data sources for critical metrics
3. Differentiators from Existing Solutions
3.1 Comparison with DeFi Llama
DeFi Llama
how3.io
Focuses primarily on TVL
Uses category-specific primary metrics
Raw data without context
Data with meaningful explanations, and perspective
Technical interface
User-friendly, jargon-free interface
Primarily for DeFi users
For all crypto investors
No valuation insights
Clear valuation assessment

3.2 Comparison with Dune Analytics
Dune Analytics
how3.io
Requires SQL knowledge, creating your own code and dashboards
No technical knowledge required - pre-compiled extensive research with AI optimized doubt-answering solution
Build-your-own dashboards
Ready-made insights
Raw on-chain data
Processed, meaningful metrics
For data analysts
For investors of all experience levels
No standardized evaluation
Consistent evaluation framework

3.3 Comparison with Traditional "DYOR" Platforms
DYOR Platforms
how3.io
Information overload
Focused, relevant insights
Inconsistent methodologies
Standardized evaluation framework
Often influenced by project marketing
Objective, data-driven assessment
Requires extensive research
Quick, comprehensive overview
No clear valuation insights
Clear undervalued/overvalued assessment

4. User Benefits
4.1 Time Efficiency
Reduce research time from hours/days to minutes
Quickly identify promising projects without technical expertise
Receive alerts about significant metric changes
Efficiently compare projects across categories
4.2 Decision Confidence
Understand why a project might be undervalued or overvalued
See how projects compare to category benchmarks
Access plain-language explanations of complex metrics
Identify sustainable projects based on revenue stability
4.3 Risk Management
Identify projects with unsustainable tokenomics
Recognize revenue fluctuations exceeding 20% quarter-over-quarter
Understand project-specific risk factors
Compare volatility across similar projects
4.4 Education
Learn what metrics matter for different types of crypto projects
Understand how to evaluate crypto fundamentals
Recognize patterns in successful projects
Build investment literacy through contextual explanations
5. Proprietary Metrics and IP
5.1 how3 Score
A proprietary composite score (0-100) incorporating:
Category-specific primary metrics
Growth trajectory (user adoption, volume)
Revenue sustainability (quarter-over-quarter stability)
Token economics health (distribution, inflation rate)
Relative valuation compared to category peers
5.2 Sustainability Index
A measure of project longevity considering:
Revenue consistency (flagging fluctuations >20% QoQ)
Excluding outlier events (airdrops, marketing campaigns)
Treasury runway and burn rate
Developer activity and commitment
User retention metrics
5.3 Real User Adoption Metric (RUAM)
A proprietary metric distinguishing between:
Genuine user activity vs. wash trading
New users vs. recycled wallets
Sustained engagement vs. airdrop farming
Value creation vs. token speculation
5.4 Fair Value Range
A dynamic valuation model providing:
Statistical fair value estimate
Upper and lower bounds for reasonable valuation
Category-specific multiple recommendations
Confidence score for valuation accuracy
6. Data Collection Strategy
6.1 Primary Data Sources
On-chain Data: Etherscan, The Graph GRT, other blockchain explorers
Market Data: CoinGecko, CoinMarketCap APIs, 
Protocol Metrics: DefiLlama, Token Terminal
Social Metrics: Social platforms APIs, sentiment analysis tools
Project Information: Official documentation, GitHub repositories
6.2 Data Processing Pipeline
Data Collection: Automated API calls and web scraping
Data Cleaning: Normalization, outlier detection, missing value handling
Metric Calculation: Derive key metrics from raw data
Benchmarking: Compare to category averages and historical trends
Insight Generation: Convert processed data into actionable insights
6.3 Update Frequency
Price Data: Real-time where possible, otherwise hourly
On-chain Metrics: Daily for most metrics
Financial Metrics: As reported, typically quarterly
Project Classifications: Monthly review, immediate update for major changes
Valuation Assessments: Daily recalculation
7. MVP Development Plan (15 Days)
Day 1-3: Project Setup and Data Integration
Set up project repository and development environment
Define database schema for project data
Establish initial API integrations for key data sources
Create data models for crypto projects
Day 4-6: Core Classification System
Implement classification system for top 50 cryptocurrencies
Develop category-specific metric calculations
Create initial valuation models for major categories
Test data processing pipeline
Day 7-9: Frontend Foundations
Develop responsive web application shell
Create project profile page template
Implement search and filter functionality
Design interactive data visualization components
Day 10-12: Integration and Testing
Connect frontend with backend data systems
Implement user authentication for premium features
Populate system with data for top 50 cryptocurrencies
Perform initial QA testing
Day 13-15: Refinement and Launch Preparation
Optimize performance and fix critical issues
Finalize monetization implementation for premium insights
Prepare documentation and help resources
Deploy MVP to production environment
8. Monetization Strategy
8.1 Freemium Model
Free Tier:


Access to basic metrics for top 5 cryptocurrencies by market cap
Basic project classifications
General market insights


Premium Tier ($99/month):


Complete access to all 300 tracked cryptocurrencies
Full historical data and trends
Advanced metrics and proprietary scores
Detailed valuation assessments
Customizable watchlists and alerts
Ability to request analysis on a cryptocurrency
8.2 Enterprise Solutions
Custom API access for institutional clients to use on their websites
White-labeled solutions for investment platforms
Specialized research reports for family offices and funds, via newsletter
9. Success Metrics
9.1 User Engagement
Active users (daily, weekly, monthly)
Average session duration
Pages viewed per session
Return visitor rate
Feature utilization rates
9.2 Business Metrics
Free-to-paid conversion rate
Monthly recurring revenue
Customer acquisition cost
Customer lifetime value
Churn rate
9.3 Product Quality Metrics
Data accuracy (compared to source of truth)
System uptime and reliability
User satisfaction scores
Net Promoter Score
Feature request fulfillment rate
10. Future Roadmap (Post-MVP)
Phase 2: Enhanced Analytics (Months 2-3)
Implement AI-driven insight generation
Add portfolio analysis tools
Develop comparative analysis features
Create customizable dashboards
Phase 3: Community Features (Months 4-5)
Implement user comments and discussions
Create analyst reputation system
Develop community-driven research initiatives
Add social sharing capabilities
Phase 4: Advanced Tools (Months 6-8)
Develop scenario analysis tools
Create investment thesis builder
Implement automated portfolio suggestions
Develop risk assessment calculator
11. Risks and Mitigations
11.1 Data Quality Risks
Risk: Inconsistent or inaccurate data from third-party sources
Mitigation: Implement data validation systems, multiple source verification, and anomaly detection
11.2 Regulatory Risks
Risk: Changing regulatory landscape around crypto investing advice
Mitigation: Clear disclaimers, legal review of content, focus on data rather than advice
11.3 Competitive Risks
Risk: Established players entering the same market segment
Mitigation: Focus on unique value proposition, rapid iteration, building loyal user base
11.4 Technical Risks
Risk: Scalability challenges with data processing
Mitigation: Cloud-based architecture, caching strategies, efficient database design
12. Conclusion
how3.io represents a paradigm shift in crypto analytics, moving from complex, technical data toward accessible, meaningful insights. By focusing on fundamental value and providing clear context around metrics that matter, we empower investors to make informed decisions in the crypto space.

