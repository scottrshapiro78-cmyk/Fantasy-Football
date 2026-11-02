"""Plain-English column explanations, added to every output workbook as a 'Column Guide'
tab and as hover comments on the header cells."""
from __future__ import annotations

import re

import openpyxl
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill

GUIDE = {
    "Overall Rk": "Where the player ranks among everyone. Burbs: by max bid. CBS: the order to draft in.",
    "Pos": "Position.",
    "Pos Rk": "Where the player ranks at his position (RB1, RB2 ...).",
    "Pos Tier": "Group of players at the position worth about the same. A new tier starts where value drops off sharply; if you miss everyone in a tier, the next tier is a real step down.",
    "Tier": "Same idea as Pos Tier but across the whole board, about one draft round each.",
    "Player": "Player (defenses show the team name).",
    "Team": "NFL team. Last season's team until free agency and the draft are loaded.",
    "Max Bid (Suggested)": "Shorthand for the two numbers: '$40 ($34)' means never pay more than $40, aim to get him for about $34.",
    "Max Bid $": "The model's dollar value for the player in The Burbs. The most he is worth; never bid past it.",
    "Suggested $": "Where to aim: what this league usually pays for this slot, but never more than 90% of the max bid.",
    "League $": "What this league typically pays for a player ranked here at his position (average of the 2025 and 2026 auctions).",
    "Bargain $": "Max bid minus League $. Positive = the model thinks he's worth more than the room usually pays (a bargain). Negative = the room usually overpays.",
    "Flag": "'Bargain' = worth at least $5 more than the usual price. 'League overpays' = the room usually pays $5+ more than he's worth; expect to let him go.",
    "PPG": "Projected fantasy points per game he plays, in this league's scoring.",
    "Exp Games (of 13)": "Games he's expected to play in your 13 regular-season games (weeks 1-14 minus a bye), after injury risk.",
    "Proj Pts (wk 1-14)": "Projected regular-season points: PPG x expected games.",
    "Repl PPG": "Points per game of a replacement-level player at the position: the kind of player left for $1 or on waivers.",
    "VORP": "Value over replacement: how many more regular-season points he gives you than a replacement player, plus a little credit for big-upside players. The engine behind every ranking and dollar value.",
    "Injury Risk (0-100)": "Chance (in %) he misses 4 or more games. Already built into Exp Games; shown for reference.",
    "Note": "'Rookie' = projected from draft slot only (no NFL history yet).",
    "nflverse ID": "Data ID used to match players. Ignore.",
    # draft-night look-back columns
    "Paid in 2026 $": "What he actually sold for in The Burbs 2026 auction.",
    "Actual Pts": "Fantasy points he has actually scored this season so far, in this league's scoring.",
    "Actual PPG": "Actual points per game played this season so far.",
    "Actual Games": "Games he has actually played this season so far.",
    "Actual Pos Rk": "His actual rank at the position this season so far, by total points.",
    "Pos Rk Diff (+ = beat)": "Projected Pos Rk minus actual. Positive = doing better than projected; negative = worse.",
    # Draft Optimizer import (the workbook's own columns)
    "Rank": "Model rank at the position (by max bid).",
    "Cap Value ($)": "The model's max bid for The Burbs (same as Max Bid $).",
    "My Assigned Bid ($)": "Your own bid. Left blank for you to fill.",
    "Value Gap ($)": "Cap Value minus your bid. Blank until you fill your bid.",
    "Proj Points": "Projected season points on the workbook's 17-game basis: PPG x expected games of 17.",
    "Games": "Expected games of 17 after injury risk, raised to the workbook's 13-game floor.",
    "Bye": "Bye week. Blank until the 2027 schedule comes out in May.",
    "ADP": "Average draft position. Blank until August.",
    "Roto Proj Pts": "Workbook formula (Rotoballer projection). Leave as is in the workbook.",
    "Blend Proj Pts": "Workbook formula (blend of your and Rotoballer projections).",
    "Disagreement %": "Workbook formula (how far apart the two projections are).",
    "Model $": "Workbook formula (Player Master price).",
    "Model - My Bid ($)": "Workbook formula.",
    "Clean Name": "Plain player name the workbook uses to match players across tabs.",
    "Roto Auction $": "Workbook formula (Rotoballer auction value).",
    "Your PPG": "Workbook formula.",
    "Roto PPG": "Workbook formula.",
    "Blend PPG": "Workbook formula.",
    "FF Guys Auction $": "FF Guys price. Blank until August.",
}


def explain(header: str) -> str | None:
    h = re.sub(r"^\d{4} ", "", str(header))
    h = re.sub(r"^Actual Pts.*", "Actual Pts", h)
    return GUIDE.get(h)


def add_guide(path) -> None:
    wb = openpyxl.load_workbook(path)
    seen = []
    for ws in wb.worksheets:
        if ws.title in ("Notes", "How to import", "Column Guide"):
            continue
        hdr_row = next((r for r in range(1, 30) if ws.cell(r, 2).value == "Player" or ws.cell(r, 1).value in GUIDE
                        or str(ws.cell(r, 1).value).endswith(" Rank")), 1)
        for c in ws[hdr_row]:
            text = explain(c.value) if c.value else None
            if text:
                c.comment = Comment(text, "Model", width=320, height=110)
                if c.value not in seen:
                    seen.append(c.value)
    if "Column Guide" in wb.sheetnames:
        del wb["Column Guide"]
    g = wb.create_sheet("Column Guide", 0 if "How to import" not in wb.sheetnames else 1)
    g.append(["Column", "What it means"])
    for c in g[1]:
        c.font, c.fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F3864")
    for h in seen:
        g.append([h, explain(h)])
    g.column_dimensions["A"].width = 26
    g.column_dimensions["B"].width = 120
    for row in g.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    wb.move_sheet(g, offset=len(wb.sheetnames) - 1 - wb.sheetnames.index("Column Guide"))  # put last
    wb.save(path)
