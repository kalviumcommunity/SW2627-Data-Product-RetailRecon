# Dashboard Architect

Welcome. SQL queries are optimized. Data is clean. Now comes the final transformation: taking insights and presenting them to humans in forms they understand. A well-designed dashboard answers business questions instantly. A poorly designed one drowns users in data. You must understand dashboard architecture - what data belongs together, how to organize information hierarchically, what metrics matter most, how to make exploration intuitive without overwhelming viewers.

Every dashboard that confused users, that had charts nobody used, that required 10 clicks to answer one question had the same root problem: it was designed by someone who understood the data but not how humans actually make decisions with information. This lesson teaches you dashboard thinking. You will learn how to organize metrics hierarchically, understand the difference between summary and detail views, design for information flow, and create dashboards that answer business questions before they are asked.

## The Real Scenario

### The Problem

A dashboard is built showing 50 charts on one page. All available. Nothing prioritized. A business user logs in, sees overwhelming chaos, gets lost scrolling, cannot find the one metric they care about. They stop using it and go back to emails and spreadsheets. The dashboard fails not because the data is wrong but because it violates how human attention works. Humans cannot process 50 things at once. They scan in order of importance. If important things are buried, they miss them.

### The Solution

Design hierarchically. Top of page: 5 KPI cards answering "are we on track?" Second section: trends and segment performance. Below: detailed drill-down available via filters. User scans from top to bottom, sees most important first, can dive deeper if needed. Attention is directed. Information is discoverable. Dashboard succeeds.

## Dashboard Hierarchy

### Information Pyramid: What Humans See First

### Level 1: Status

KPI summary cards (5 cards max)

Current month revenue, active users, churn rate. Answer: "Are we on track?" Scans in 5 seconds. If red, user investigates. If green, confidence increases.

### Level 2: Trends

Time series charts (revenue/month, churn/month)

Answer: "Is it getting better or worse?" Scrolls to see trends. Identifies patterns. Notices when something changed direction.

### Level 3: Segments

Revenue by customer type, churn by segment

Answer: "Which parts of the business need attention?" Continues scrolling if needed. Finds patterns in segments.

### Level 4: Detail

Filters, drill-down, raw data export

Answer: "I found an anomaly, now show me everything." Only power users reach here. Detail available when needed.

You just learned the information hierarchy that makes dashboards usable. Summary first. Trends second. Detail third. Now apply this structure.

## Designing for Human Attention

### Organizing Information for Discovery

### Principle 1: Progressive Disclosure

Show summary immediately. Hide detail behind filters. User decides "do I need to explore deeper?" If KPIs are green, they leave satisfied. If one is red, they filter that segment and zoom in. Information appears as needed, not all at once. Cognitive load is managed.

### Principle 2: Spatial Organization

Top-left is most visible (western reading pattern). Place most critical KPI there. Top row holds all critical metrics. Left side for dimensions that matter most (product vs region). Right side for secondary metrics. Bottom for exploratory detail. Eyes naturally flow left-to-right, top-to-bottom. Use that flow.

### Principle 3: Consistent Metaphor

If red means "problem" in one KPI card, red means problem everywhere. If up arrow means "good", up means good for all trends. Consistent visual language reduces cognitive load. User learns the rules once, applies everywhere.

### Principle 4: Context Over Numbers

Show "Revenue:50k". Context (comparison, trend, target) makes numbers meaningful. A reader does not have to know "is44k, so yes, $50k is good. Include context always.

You just learned four design principles that make dashboards actually usable. Now you will implement them in code across the next lessons. Dashboard architecture is your foundation.

## Bonus Resources

- [Dashboard Design Best Practices - principles for information architecture and user attention flow](https://en.wikipedia.org/wiki/Dashboard_(business))
- [Stephen Few's Perceptual Edge - foundational work on data visualization design and effective communication](https://www.perceptualedge.com/articles/Whitepapers/Designing_Dashboards.pdf)
- [Information Hierarchy in UI Design - how to organize complex information for human processing](https://www.interaction-design.org/literature/article/information-architecture)