from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
    Spacer, Image, Table, TableStyle, KeepTogether, PageBreak, FrameBreak, NextPageTemplate)
from pathlib import Path

OUT=Path('output/pdf/munchkin_rl_report.pdf')
OUT.parent.mkdir(parents=True,exist_ok=True)
W,H=letter
M=.48*inch; GAP=.22*inch; CW=(W-2*M-GAP)/2
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Title2', parent=styles['Title'], fontName='Helvetica-Bold',fontSize=16,leading=17,alignment=TA_CENTER,textColor=colors.HexColor('#15324b'),spaceAfter=4))
styles.add(ParagraphStyle(name='Authors',parent=styles['Normal'],fontSize=7.2,leading=8.5,alignment=TA_CENTER,spaceAfter=1))
styles.add(ParagraphStyle(name='Abstract',parent=styles['Normal'],fontSize=7.6,leading=9,alignment=TA_LEFT,spaceAfter=4,borderColor=colors.HexColor('#ccd7df'),borderWidth=.4,borderPadding=4))
styles.add(ParagraphStyle(name='H2x',parent=styles['Heading2'],fontSize=10.2,leading=11,textColor=colors.HexColor('#15324b'),spaceBefore=4,spaceAfter=2,keepWithNext=True))
styles.add(ParagraphStyle(name='Bodyx',parent=styles['BodyText'],fontSize=8.0,leading=9.7,spaceAfter=3,allowWidows=0,allowOrphans=0))
styles.add(ParagraphStyle(name='Smallx',parent=styles['BodyText'],fontSize=6.8,leading=8.0,spaceAfter=2))
styles.add(ParagraphStyle(name='Captionx',parent=styles['BodyText'],fontSize=6.7,leading=7.8,spaceBefore=1,spaceAfter=3))
styles.add(ParagraphStyle(name='Eq',parent=styles['BodyText'],fontName='Helvetica-Oblique',fontSize=8,leading=10,leftIndent=4,rightIndent=4,spaceBefore=1,spaceAfter=3))

class TwoColDoc(BaseDocTemplate):
    def __init__(self,path,**kwargs):
        super().__init__(path,**kwargs)
        bottom=.52*inch
        left1=Frame(M,bottom,CW,H-2.58*inch,id='left1',leftPadding=0,rightPadding=4,topPadding=0,bottomPadding=0)
        right1=Frame(M+CW+GAP,bottom,CW,H-2.58*inch,id='right1',leftPadding=4,rightPadding=0,topPadding=0,bottomPadding=0)
        head=Frame(M,H-2.48*inch,W-2*M,1.96*inch,id='head',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        left=Frame(M,bottom,CW,H-1.02*inch,id='left',leftPadding=0,rightPadding=4,topPadding=0,bottomPadding=0)
        right=Frame(M+CW+GAP,bottom,CW,H-1.02*inch,id='right',leftPadding=4,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates([PageTemplate(id='first',frames=[head,left1,right1],onPage=self.footer),PageTemplate(id='regular',frames=[left,right],onPage=self.footer)])
    def footer(self,canvas,doc):
        canvas.saveState(); canvas.setFont('Helvetica',6.5); canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawCentredString(W/2,.24*inch,f'Munchkin • {doc.page}')
        canvas.restoreState()

def P(s,style='Bodyx'): return Paragraph(s,styles[style])
story=[]
story += [P('Munchkin: Reinforcement Learning for Dynamic Food-Delivery Dispatch','Title2'),
          P('<b>Anshit Agarwal · Namya Dhingra</b>','Authors'),
          P('GitHub: <link href="https://github.com/Anshit2005/Munchkin---RL-Based-Dynamic-Food-Delivery-Dispatch-System" color="#1769aa">github.com/Anshit2005/Munchkin---RL-Based-Dynamic-Food-Delivery-Dispatch-System</link>','Authors'),
          P('<b>Abstract—</b> We present Munchkin, a minute-stepped food-delivery dispatch simulator and comparative study of three heuristics, Deep Q-Network (DQN), and Proximal Policy Optimization (PPO). A fixed 60-minute episode couples stochastic order arrivals, restaurant preparation, travel, deadlines, and finite driver capacity. The observation contains 1,849 features and the masked discrete action space contains 1,025 actions. Across 11 five-seed scenario configurations, heuristics remain highly competitive; learned policies show useful behavior in selected conditions but do not dominate consistently.','Abstract'),
          FrameBreak(),NextPageTemplate('regular'),
          P('1. Problem and environment','H2x'),
          P('Munchkin models online assignment of food orders to delivery partners. Each episode simulates one hour: one real-time second corresponds to one game minute. The map is 100×70 units, with 3–8 restaurants and 3–16 drivers during training. New orders follow a Poisson process with scenario rate λ∈[1,8] orders/minute; half the configured rate is seeded as initial backlog. Preparation time is a clipped normal sample (mean 0.5, SD 0.17 min, bounds 0.15–1.0); estimates are log-normal noisy (σ=0.2). Deadlines equal estimated preparation plus nominal travel time and five minutes of slack. Traffic uses light/normal/heavy multipliers 1.2/1.0/0.7 with minute-level noise. Drivers move, wait for food, and deliver as time advances; unfinished orders incur a terminal penalty.'),
          P('2. RL formulation','H2x'),
          P('<b>State space.</b> At each minute the fixed-size observation s<sub>t</sub>∈R<super>1849</super> concatenates 16 driver records (11 values each), 64 order records (10 each), a 16×64 driver–order ETA matrix, and 9 global features: 16(11)+64(10)+16(64)+9=1849. Driver features encode presence, position, availability/travel state, destination, remaining distance/time, and food-wait status. Order features encode presence, restaurant/customer coordinates, age, preparation remaining, restaurant load, route length, and deadline slack. Global values encode normalized time/horizon, dispatchable/pending/available counts, demand rate, restaurant count, overflow, and traffic ratio. Padding fixes dimensions; masks exclude absent or infeasible pairs.'),
          FrameBreak(),
          P('<b>Action space.</b> Discrete dispatch driver i to order j, encoded 64i+j for i&lt;16,j&lt;64, or WAIT (index 1024): |A|=1,025. Only feasible pairs are enabled. Assignment consumes no simulated minute, allowing multiple dispatches before WAIT advances the clock.'),
          P('<b>Reward.</b> Shared shaped reward per step:','Bodyx'),
          P('r<sub>t</sub> = 10D<sub>t</sub> − 2W<sub>t</sub> − 4L<sub>t</sub> − 0.1M<sub>t</sub> − I<sub>t</sub> − 5F<sub>t</sub>·1[t=T].','Eq'),
          P('D = new deliveries; W = pending orders after movement; L = overdue pending; M = distance moved; I=min(available drivers, dispatchable unassigned orders) (idle-capacity cost); F = unfinished pending orders at the 60-minute cutoff. Game score is separately 10 points per delivery; it is not the shaped RL return.'),
          P('3. Policies and training','H2x'),
          P('Baselines are Nearest Driver, Oldest/Most Urgent (least deadline slack), and Minimum Estimated Delivery Time. DQN and PPO share state, reward, and action masks. DQN uses pairwise Q values, masked Double-DQN targets, uniform replay (capacity 20,000; batch 128), γ=0.99, learning rate 3e-4, warm-up 32 transitions, updates every four steps, target sync every 1,000 steps, and ε decay 1.0→0.05. PPO uses masked categorical actions, clipped objective ε=0.2, γ=0.99, GAE λ=0.95, 8 epochs, minibatches of 128, and learning rate 3e-4. PPO samples during training and uses deterministic masked argmax at evaluation. Both trained for 500 episodes over randomized demand, fleet, restaurant count, and traffic. Best checkpoints: DQN episode 175 (validation shaped return −3,949.3) and PPO episode 300 (−4,806.9). The sweep uses five matched seeds per cell (11 configurations; 275 policy-episodes total), each 60 minutes.'),
          PageBreak(),
          Image('reports/munchkin_rl_report/images/results.png',width=CW,height=CW*.34),
          P('Fig. 1. Results across demand, driver capacity, and training. Error bars show across-seed SD for completed orders; training uses a 25-episode moving mean of game points.','Captionx'),
          P('4. Results and discussion','H2x')]

data=[['Policy','Low','Medium','High'],['Nearest','121.2 / 1212','307.4 / 3074','437.8 / 4378'],['Urgent','121.0 / 1210','305.0 / 3050','356.2 / 3562'],['Min ETA','121.2 / 1212','307.8 / 3078','452.4 / 4524'],['DQN','120.6 / 1206','306.8 / 3068','444.4 / 4444'],['PPO','121.2 / 1212','308.6 / 3086','440.8 / 4408']]
t=Table(data,colWidths=[.63*inch,.72*inch,.78*inch,.72*inch],repeatRows=1,hAlign='LEFT')
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#15324b')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),5.7),('LEADING',(0,0),(-1,-1),6.7),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#cad4dc')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f6f9')]),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
story += [FrameBreak(),P('Table 1. Mean completed orders / game points by demand under normal traffic (five seeds).','Captionx'),t,Spacer(1,2),
P('At low and medium demand, methods are close; PPO leads medium-demand points (3,086), with Minimum ETA at 3,078. At high demand, Minimum ETA completes most orders (452.4); DQN completes 444.4 but has the best mean shaped return (−1,184 versus −1,681 for Minimum ETA). Thus game score and shaped reward rank policies differently. With medium demand and three drivers, Minimum ETA completes 179.4 versus DQN 171.0 and PPO 175.0. In the severe high-demand/heavy-traffic case with three restaurants and three drivers, Minimum ETA completes 177.6, PPO 144.6, and DQN 100.4. These results do not show RL universally outperforming heuristics; the constrained-fleet case exposes a weakness of the learned dispatchers.'),
P('<b>Conclusion.</b> Munchkin provides a reproducible benchmark with explicit state/action masks and shared scenario dynamics. Current learned policies are competitive in selected regimes but not uniformly superior. Future work should target constrained-fleet credit assignment, broader reward/hyperparameter sweeps, and more seeds before statistical conclusions.'),
P('References','H2x'),
P('[1] V. Mnih et al., “Human-level control through deep reinforcement learning,” <i>Nature</i>, 518, 529–533, 2015. doi:10.1038/nature14236.<br/>[2] J. Schulman et al., “Proximal Policy Optimization Algorithms,” arXiv:1707.06347, 2017.<br/>[3] Munchkin project repository and simulator implementation, GitHub, 2026.','Smallx')]

doc=TwoColDoc(str(OUT),pagesize=letter,title='Munchkin RL Dispatch Report',author='Anshit Agarwal; Namya Dhingra')
doc.build(story)
print(OUT)
