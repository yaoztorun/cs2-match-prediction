# Unresolved Team Name Review

Deep-dive on the 25 `unresolved` names from `data/interim/team_aliases.csv` (Phase 2 team-identity pass). **No identity is split or merged here** - this is a proposal for human review only; `team_aliases.csv` is not modified by this script. Per explicit guidance for this pass: for established professional organization names, zero roster overlap between two team_id instances is *not*, by itself, sufficient evidence that they are different organizations - full roster turnover is expected for real orgs over a multi-year dataset. That is reflected in the proposed decisions below.

**Proposed decision categories** (proposed only, not applied): `KEEP_AS_SINGLE_TEAM`, `NEEDS_EPISODE_SPLIT`, `EXCLUDE_FROM_IDENTITY_DEPENDENT_FEATURES`, `MANUAL_REVIEW`.

## `9INE`

- First appearance: 2023-10-06 20:00:00  |  Last appearance: 2026-06-25 15:30:00
- Matches: 206  |  Tournaments (58): ['BC.Game Masters Championship', 'BLAST Bounty Winter 2026. Closed Qualifiers', 'BLAST Rising Europe Spring 2025', 'BLAST Rising Europe Spring 2025. Closed Qualifier', 'BetBoom Dacha. Closed Qualifier', 'Birch Cup 2025', 'CCT North Europe Series #8', 'CCT Season 2 European Series #10', 'CCT Season 2 European Series #15', 'CCT Season 2 European Series #16', 'CCT Season 2 European Series #17', 'CCT Season 2 European Series #18', 'CCT Season 2 European Series #7', 'CCT Season 3 European Series #1', 'CCT Season 3 European Series #12', 'CCT Season 3 European Series #14', 'CCT Season 3 European Series #3', 'CCT Season 3 European Series #4', 'Conquest of Prague 2025', 'DraculaN #1', 'ESL Challenger League Season 46 — Europe', 'ESL Challenger League Season 47 Relegation: Europe', 'ESL Challenger League Season 47 — Europe', 'ESL Challenger League Season 50: Europe — Cup #1', 'ESL Challenger League Season 50: Europe — Cup #2', 'ESL Challenger League Season 51: Europe — Cup #4', 'ESL Pro League Season 22. European Qualifier', 'Esports World Cup 2024 по CS2. Open Qualifier', 'European Pro League Season 20', 'Exort The Proving Grounds Season 3', 'Exort The Proving Grounds Season 5', 'FRAG Miami 2 2026', 'Galaxy Battle 2025 // Phase 5', 'IEM Dallas 2024: Qualifier Europe', 'IEM Rio 2024: Qualifier Europe', 'IEM Rio 2026. Closed Qualifier', 'IstanbuLAN 2026', 'NODWIN Clutch Series #4', 'PGL Astana 2025. Closed European Qualifier', 'PGL Bucharest 2025. European Qualifiers', 'PGL Bucharest 2026. European Qualifier', 'PGL CS2 Major Copenhagen 2024: Open Qualifiers', 'Parken Challenger Championship #1', 'RES European Series #5', 'Regional Clash Arena Europe', 'Roman Imperium Cup VI', 'Roobet Cup 2023', 'Stake Ranked Episode 1', 'StarLadder StarSeries Fall 2025', 'StarLadder StarSeries Fall 2025. Closed Qualifier', 'Super DraculaN Season 1', 'Thunderpick World Championship 2024 - Qualifiers Europe #2', 'Thunderpick World Championship 2025: European Series #1', 'YaLLa Compass Fall 2023', 'YaLLa Compass Fall 2024', 'YaLLa Compass Spring 2024', 'YaLLa Compass Summer 2024', 'YaLLa Compass Winter 2025']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['1626', '1627', '789', '948', '988'] | 2023-10-06 20:00:00 | 2023-11-01 21:00:00 | ['232905', '232979', '232975', '233315', '233399', '233612', '233291', '233384', '233387', '233521', '233297'] |
| 2 | ['1626', '1627', '724', '789', '948'] | 2023-11-07 21:00:00 | 2023-11-08 21:00:00 | ['233926', '234026'] |
| 3 | ['1626', '1627', '789', '948', '988'] | 2023-11-16 14:55:00 | 2023-11-20 18:15:00 | ['234163', '234170', '234330'] |
| 4 | ['1626', '1627', '6715', '789', '948'] | 2023-11-21 17:55:00 | 2023-11-21 17:55:00 | ['234336'] |
| 5 | ['1626', '1627', '724', '789', '948'] | 2023-12-05 21:00:00 | 2023-12-06 21:00:00 | ['235049', '235061'] |
| 6 | ['1626', '1627', '6715', '789', '948'] | 2023-12-12 12:00:00 | 2023-12-13 21:00:00 | ['235565', '235572', '235575'] |
| 7 | ['1626', '3198', '949', '988'] | 2024-01-09 18:00:00 | 2024-02-26 15:50:00 | ['236254', '236319', '236326', '236339', '236786', '238818', '239054', '239335'] |
| 8 | ['1626'] | 2024-02-27 22:00:00 | 2024-02-27 22:00:00 | ['240011'] |
| 9 | ['1626', '3198', '949', '988'] | 2024-03-03 20:00:00 | 2024-03-03 20:00:00 | ['240424'] |
| 10 | ['1626', '3198', '724', '988'] | 2024-03-06 22:00:00 | 2024-04-03 21:00:00 | ['240017', '240040', '240026', '240054'] |
| 11 | ['1626', '3198', '949', '988'] | 2024-04-17 22:00:00 | 2024-04-17 22:00:00 | ['243756'] |
| 12 | ['1626', '3198', '724', '988'] | 2024-04-20 16:00:00 | 2024-05-15 16:30:00 | ['243161', '243906', '244752'] |
| 13 | ['1486', '1575', '2991', '436', '488'] | 2024-06-05 21:00:00 | 2024-06-08 21:00:00 | ['248357', '248359', '248362'] |
| 14 | ['3198', '3314', '724'] | 2024-06-11 20:00:00 | 2024-06-12 20:00:00 | ['249130', '249138'] |
| 15 | ['1486', '1575', '2991', '436', '488'] | 2024-06-13 18:40:00 | 2024-06-13 18:40:00 | ['249098'] |
| 16 | ['3198', '3314', '724'] | 2024-06-13 21:00:00 | 2024-06-15 20:00:00 | ['249148', '249151', '249162'] |
| 17 | ['1486', '1575', '2991', '436', '488'] | 2024-08-04 22:10:00 | 2024-12-15 13:30:00 | ['252191', '253100', '253209', '253230', '253451', '254058', '254003', '254079', '254332', '254364', '254253', '254528', '254553', '254698', '254807', '255061', '255222', '255380', '258127', '258132', '258395', '258399', '258168', '258427', '258210', '258443', '258451', '258455', '260108', '260306', '260354', '260416', '260721', '260739', '260783'] |
| 18 | ['1575', '3241', '3904', '436', '483'] | 2025-01-15 12:00:00 | 2025-06-19 14:00:00 | ['262246', '262922', '262954', '263246', '263436', '263578', '263803', '264018', '263917', '263946', '263952', '263951', '264280', '264814', '264358', '264871', '265005', '265124', '265335', '265494', '266051', '266065', '266069', '266072', '266478', '266075', '266063', '267245', '267305', '267313', '267316', '269822', '269940', '270076', '270152', '270380', '270641', '270643', '271558', '271561', '271565', '272800'] |
| 19 | ['27834', '3241', '436', '483', '6721'] | 2025-07-14 18:00:00 | 2025-07-14 18:00:00 | ['273775'] |
| 20 | ['1316', '3241', '436', '483', '6721'] | 2025-07-29 20:05:00 | 2025-08-13 11:00:00 | ['274390', '274465', '274588', '274950', '275259', '275371', '275584', '275726', '275863'] |
| 21 | ['1316', '1575', '436', '483', '6721'] | 2025-08-13 16:00:00 | 2025-08-13 19:40:00 | ['275987', '275989'] |
| 22 | ['1316', '3241', '436', '483', '6721'] | 2025-08-14 20:00:00 | 2025-08-16 20:00:00 | ['275833', '276077', '276158', '276097'] |
| 23 | ['1316', '1575', '436', '483', '6721'] | 2025-08-17 16:00:00 | 2025-08-17 16:00:00 | ['276212'] |
| 24 | ['1316', '3241', '436', '483', '6721'] | 2025-08-18 14:00:00 | 2025-08-27 20:00:00 | ['276244', '276411', '276211'] |
| 25 | ['1575', '3241', '3904', '436', '483'] | 2025-08-29 20:40:00 | 2025-08-30 17:00:00 | ['276691', '276702', '276707'] |
| 26 | ['1316', '3241', '436', '483', '6721'] | 2025-08-30 20:00:00 | 2025-08-30 20:00:00 | ['276857'] |
| 27 | ['1575', '3241', '3904', '436', '483'] | 2025-08-30 21:00:00 | 2025-08-30 21:00:00 | ['276709'] |
| 28 | ['1316', '3241', '436', '483', '6721'] | 2025-08-31 20:00:00 | 2025-08-31 20:00:00 | ['276908'] |
| 29 | ['3241', '436', '483', '6721'] | 2025-09-09 20:00:00 | 2025-09-18 20:00:00 | ['277151', '277423', '277650', '277732'] |
| 30 | ['3241', '436', '4825', '483', '6721'] | 2025-09-18 21:10:00 | 2025-09-19 15:00:00 | ['277165', '277814'] |
| 31 | ['3241', '436', '483', '6721'] | 2025-09-19 20:00:00 | 2025-09-20 20:00:00 | ['277740', '277744'] |
| 32 | ['3241', '436', '4825', '483', '6721'] | 2025-09-28 10:00:00 | 2025-10-12 20:00:00 | ['278842', '278848', '278851', '279860', '279862'] |
| 33 | ['1316', '3235', '3241', '436', '6721'] | 2025-12-06 12:00:00 | 2025-12-14 17:30:00 | ['282794', '283226', '283280', '283282'] |
| 34 | ['2814', '3235', '3241', '436', '6721'] | 2026-01-04 18:00:00 | 2026-02-04 18:00:00 | ['283792', '283797', '283799', '283611', '284768', '284791', '284798', '284814', '284820', '286044', '286248'] |
| 35 | ['3235', '3241', '436', '6721'] | 2026-02-04 21:30:00 | 2026-02-05 15:30:00 | ['286268', '286302'] |
| 36 | ['2814', '3235', '3241', '436', '6721'] | 2026-02-12 18:30:00 | 2026-02-28 12:00:00 | ['286856', '286868', '286874', '287302', '287305', '287311', '287316', '287323', '287420', '287428', '287438'] |
| 37 | ['3235', '3241', '436', '5509', '6721'] | 2026-03-12 13:00:00 | 2026-04-23 20:00:00 | ['288228', '288231', '288234', '288241', '288504', '289024', '289271', '289280', '289285', '292024'] |
| 38 | ['2731', '436', '5509', '6721'] | 2026-06-23 15:00:00 | 2026-06-25 15:30:00 | ['295794', '295803', '295831'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['1626', '3198', '949', '988'] (active 2024-04-17 22:00:00 to 2024-04-17 22:00:00) vs roster ['1486', '1575', '2991', '436', '488'] (active 2024-06-05 21:00:00 to 2024-06-08 21:00:00) (48-day gap)
- roster ['1626', '3198', '949', '988'] (active 2024-04-17 22:00:00 to 2024-04-17 22:00:00) vs roster ['1486', '1575', '2991', '436', '488'] (active 2024-06-13 18:40:00 to 2024-06-13 18:40:00) (56-day gap)
- roster ['1626', '3198', '724', '988'] (active 2024-04-20 16:00:00 to 2024-05-15 16:30:00) vs roster ['1486', '1575', '2991', '436', '488'] (active 2024-06-05 21:00:00 to 2024-06-08 21:00:00) (21-day gap)
- roster ['1626', '3198', '724', '988'] (active 2024-04-20 16:00:00 to 2024-05-15 16:30:00) vs roster ['1486', '1575', '2991', '436', '488'] (active 2024-06-13 18:40:00 to 2024-06-13 18:40:00) (29-day gap)
- roster ['1486', '1575', '2991', '436', '488'] (active 2024-06-05 21:00:00 to 2024-06-08 21:00:00) vs roster ['3198', '3314', '724'] (active 2024-06-11 20:00:00 to 2024-06-12 20:00:00) (2-day gap)
- roster ['1486', '1575', '2991', '436', '488'] (active 2024-06-05 21:00:00 to 2024-06-08 21:00:00) vs roster ['3198', '3314', '724'] (active 2024-06-13 21:00:00 to 2024-06-15 20:00:00) (5-day gap)
- roster ['3198', '3314', '724'] (active 2024-06-11 20:00:00 to 2024-06-12 20:00:00) vs roster ['1486', '1575', '2991', '436', '488'] (active 2024-06-13 18:40:00 to 2024-06-13 18:40:00) (0-day gap)
- roster ['3198', '3314', '724'] (active 2024-06-11 20:00:00 to 2024-06-12 20:00:00) vs roster ['1486', '1575', '2991', '436', '488'] (active 2024-08-04 22:10:00 to 2024-12-15 13:30:00) (53-day gap)
- roster ['1486', '1575', '2991', '436', '488'] (active 2024-06-13 18:40:00 to 2024-06-13 18:40:00) vs roster ['3198', '3314', '724'] (active 2024-06-13 21:00:00 to 2024-06-15 20:00:00) (0-day gap)
- roster ['3198', '3314', '724'] (active 2024-06-13 21:00:00 to 2024-06-15 20:00:00) vs roster ['1486', '1575', '2991', '436', '488'] (active 2024-08-04 22:10:00 to 2024-12-15 13:30:00) (50-day gap)

**Assessment**: (a) same organization with roster turnover
**Proposed decision**: `KEEP_AS_SINGLE_TEAM`
**Rationale**: 206 matches under this name with clean sequential roster eras (each era's appearances are chronologically disjoint from the next - a roster hand-off pattern, not simultaneous usage). Per the review's own rule, zero overlap between two eras of an established, high-volume name is NOT by itself evidence of a different organization - full roster turnover over a multi-year dataset is expected for real orgs. Recommend keeping as a single identity, with the era boundaries available for anyone who later wants roster-level (not org-level) granularity.

## `AMKAL Esports`

- First appearance: 2024-01-18 00:05:00  |  Last appearance: 2025-11-30 15:00:00
- Matches: 126  |  Tournaments (35): ['BLAST Premier: Fall Showdown 2024', 'BetBoom Dacha Belgrade 2024 #2. European Qualifier', 'BetBoom Dacha CS2 Belgrade 2024: Qualifier Europe', 'CCT Global Finals', 'CCT Season 2 European Series #1', 'CCT Season 2 European Series #15', 'CCT Season 2 European Series #16', 'CCT Season 2 European Series #4', 'CCT Season 2 European Series #7', 'CCT Season 2 European Series #8', 'CCT Season 3 European Series #10', 'CCT Season 3 European Series #3', 'CCT Season 3 European Series #7', 'Esports World Cup 2024 по CS2. Open Qualifier', 'European Pro League Series 2', 'European Pro League Series 3', 'IEM Chengdu 2024. Closed Qualifier', 'IEM Chengdu 2024. Open Qualifier', 'IEM Dallas 2024: Qualifier Europe', 'IEM Rio 2024: Qualifier Europe', 'Majestic LanDaLan #3. Closed Qualifier', 'PGL Astana 2025. Closed European Qualifier', 'PGL Astana 2025. Open Qualifier', 'PGL CS2 Major Copenhagen 2024: Closed Qualifiers', 'PGL CS2 Major Copenhagen 2024: European RMR A', 'PGL Major Copenhagen 2024', 'Perfect World Shanghai Major 2024: EU Qualifier', 'RES European Masters Fall 2024', 'RES European Series #1', 'Skyesports Championship 2024', 'Thunderpick World Championship 2024 - Qualifiers Europe #1', 'Thunderpick World Championship 2024 - Qualifiers Europe #2', 'YaLLa Compass Fall 2024', 'YaLLa Compass Spring 2024', 'YaLLa Compass Summer 2024']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['1436', '3464', '5156', '932', '961'] | 2024-01-18 00:05:00 | 2024-03-18 16:55:00 | ['237487', '237486', '237502', '237459', '237687', '237741', '239234', '239242', '239250', '239251', '239195', '239522', '239546', '239561', '239662', '240243', '240246', '240593', '240623', '240550', '241644', '241672'] |
| 2 | ['1436', '3464', '5156', '5310', '961'] | 2024-04-02 12:30:00 | 2024-06-09 21:00:00 | ['242773', '242811', '242814', '242739', '242816', '242820', '243096', '242857', '242863', '242867', '243502', '242781', '243744', '243606', '243961', '243998', '244228', '244235', '244243', '244327', '244350', '244353', '244359', '244624', '244629', '244632', '245022', '248226', '248230', '248212', '248217', '248939', '248941'] |
| 3 | ['1436', '3464', '5310', '5948', '961'] | 2024-07-24 13:25:00 | 2024-07-27 15:00:00 | ['251287', '251295', '251305', '251318', '250773'] |
| 4 | ['1436', '3464', '5310', '961'] | 2024-07-31 18:00:00 | 2024-08-16 21:20:00 | ['251592', '251596', '252200', '253196'] |
| 5 | ['1436', '3464', '7528', '961'] | 2024-08-18 23:35:00 | 2024-08-21 20:00:00 | ['253467', '253465', '253074', '253650'] |
| 6 | ['1436', '3464', '960', '961'] | 2024-08-21 22:00:00 | 2024-08-21 22:00:00 | ['252991'] |
| 7 | ['1436', '3464', '7528', '961'] | 2024-08-22 18:00:00 | 2024-08-22 18:00:00 | ['253726'] |
| 8 | ['1436', '3464', '960', '961'] | 2024-08-24 00:45:00 | 2024-08-24 00:45:00 | ['253003'] |
| 9 | ['1436', '3464', '7528', '961'] | 2024-08-28 16:30:00 | 2024-12-14 21:00:00 | ['254006', '254084', '254349', '254538', '255079', '255226', '255591', '255593', '258392', '258411', '258421', '258438', '258450', '260704', '260703'] |
| 10 | ['26759', '4922', '4923', '5451', '7929'] | 2025-01-21 15:00:00 | 2025-01-29 00:00:00 | ['262931', '263241', '263451', '263467', '263531'] |
| 11 | ['26759', '4922', '4923', '5241', '7929'] | 2025-03-15 19:15:00 | 2025-03-30 14:00:00 | ['266688', '266687', '266696', '267259', '267340', '267343', '267349', '267351'] |
| 12 | ['4922', '4923', '5241', '7929'] | 2025-06-10 20:00:00 | 2025-06-20 17:00:00 | ['272473', '272483', '272519', '272639', '272805'] |
| 13 | ['1436', '2163', '5241', '611', '966'] | 2025-08-20 20:00:00 | 2025-09-28 11:30:00 | ['276473', '276561', '276823', '276873', '276941', '277177', '277596', '277961', '278019', '278249', '278451', '278636', '278883'] |
| 14 | ['1436', '2070', '2163', '25998', '966'] | 2025-10-29 21:40:00 | 2025-11-30 15:00:00 | ['280669', '280739', '280819', '280983', '281093', '282212', '282216', '282221'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['1436', '3464', '7528', '961'] (active 2024-08-28 16:30:00 to 2024-12-14 21:00:00) vs roster ['26759', '4922', '4923', '5451', '7929'] (active 2025-01-21 15:00:00 to 2025-01-29 00:00:00) (37-day gap)

**Assessment**: (a) same organization with roster turnover
**Proposed decision**: `KEEP_AS_SINGLE_TEAM`
**Rationale**: 126 matches under this name with clean sequential roster eras (each era's appearances are chronologically disjoint from the next - a roster hand-off pattern, not simultaneous usage). Per the review's own rule, zero overlap between two eras of an established, high-volume name is NOT by itself evidence of a different organization - full roster turnover over a multi-year dataset is expected for real orgs. Recommend keeping as a single identity, with the era boundaries available for anyone who later wants roster-level (not org-level) granularity.

## `Aurora Gaming`

- First appearance: 2023-10-06 20:00:00  |  Last appearance: 2026-06-20 16:45:00
- Matches: 267  |  Tournaments (67): ['BLAST Bounty Fall 2025', 'BLAST Bounty Fall 2025. Closed Qualifiers', 'BLAST Bounty Winter 2026. Closed Qualifiers', 'BLAST Open Spring 2026', 'BLAST.tv Austin Major 2025', 'BetBoom Dacha Belgrade 2024 #2', 'BetBoom Dacha CS2 Belgrade 2024', 'BetBoom Dacha CS2 Belgrade 2024: Qualifier Europe', 'BetBoom Dacha. Closed Qualifier', 'BetBoom LanDaLan #2. Closed Qualifier', 'CCT Central Europe Series #8', 'CCT Global Finals', 'CCT Online Finals #4', 'CCT Online Finals #5', 'CCT Season 2 European Series #12', 'CCT Season 2 European Series #13', 'CCT Season 2 European Series #14', 'CCT Season 2 European Series #16', 'CCT Season 2 European Series #2', 'DraculaN Season 6', 'ESL Challenger Atlanta 2023. Qualifiers', 'ESL Challenger Atlanta 2024. Qualifier Europe', 'ESL Challenger Jonköping 2023. Qualifiers', 'ESL Challenger Jonköping 2024', 'ESL Challenger Jonköping 2024. Closed Qualifier', 'ESL Challenger Jonköping 2024. Open Qualifier', 'ESL Challenger League Season 46 — Europe', 'ESL Challenger League Season 47 — Europe', 'ESL Challenger League Season 48 — Europe', 'ESL Challenger League Season 49 — Europe', 'ESL Challenger Melbourne 2024', 'ESL Challenger Melbourne 2024 - Closed Qualifiers', 'ESL Pro League Season 20: European Conference', 'ESL Pro League Season 22', 'ESL Pro League Season 23', 'Esports World Cup 2024 по CS2. Open Qualifier', 'Esports World Cup 2025 по CS2', 'FISSURE PLAYGROUND 2 — CS', 'IEM Cologne 2025', 'IEM Cologne Major 2026', 'IEM Dallas 2025', 'IEM Dallas 2025: Qualifier Europe', 'IEM Krakow 2026', 'IEM Rio 2026', 'MESA Nomadic Masters Spring 2024', 'PARI, PLEASE', 'PGL Astana 2025', 'PGL Astana 2025. Open Qualifier', 'PGL Astana 2026', 'PGL Bucharest 2025', 'PGL CS2 Major Copenhagen 2024: Closed Qualifiers', 'PGL Cluj-Napoca 2026', 'PGL Masters Bucharest 2025', 'Perfect World Shanghai Major 2024: European RMR B', 'RES Eastern European Masters: Spring 2024', 'RES European Masters Fall 2024', 'RES European Series #1', 'RES Regional Champions', 'Roobet Cup 2023', 'Skyesports Championship 2024', 'Skyesports Masters 2024', 'StarLadder Budapest Major 2025', 'Thunderpick World Championship 2024', 'Thunderpick World Championship 2025', 'YaLLa Compass Fall 2023', 'YaLLa Compass Summer 2024', 'YaLLa Compass Winter 2025']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['1008', '1867', '1892', '2070', '5499'] | 2023-10-06 20:00:00 | 2023-11-07 21:00:00 | ['232897', '232913', '232924', '232928', '232931', '232959', '232953', '233269', '233372', '233511', '233514', '233764', '233274', '233769', '234069', '233937'] |
| 2 | ['1008', '1867', '1892', '2070', '8056'] | 2023-11-08 15:30:00 | 2023-11-08 15:30:00 | ['234071'] |
| 3 | ['1008', '1867', '1892', '2070', '3059'] | 2023-11-08 21:00:00 | 2024-01-19 22:00:00 | ['234042', '234155', '234249', '234159', '234259', '234659', '234472', '234665', '234675', '234476', '234478', '234333', '235183', '235308', '235342', '235377', '235383', '235380', '235569', '235574', '237465', '237693', '237745'] |
| 4 | ['1008', '1892', '2070', '3059'] | 2024-02-17 17:00:00 | 2024-02-17 20:35:00 | ['239679', '239682', '239689'] |
| 5 | ['1008', '1892', '2070', '2988', '3059'] | 2024-02-18 16:00:00 | 2024-04-01 21:00:00 | ['239686', '239691', '240288', '240298', '240262', '240304', '240307', '240310', '240312', '240267', '240479', '239980', '239955', '239962', '242503', '242620', '242636', '239988', '240001'] |
| 6 | ['1008', '1892', '2070', '3059', '4543'] | 2024-04-02 16:30:00 | 2024-04-09 17:00:00 | ['242667', '242678', '242681', '242683', '242834', '243069', '242837', '242842', '242844'] |
| 7 | ['1008', '1892', '2070', '2988', '3059'] | 2024-04-09 21:00:00 | 2024-04-09 21:00:00 | ['243144'] |
| 8 | ['1008', '1892', '2070', '3059', '4543'] | 2024-04-10 18:00:00 | 2024-04-10 18:00:00 | ['242860'] |
| 9 | ['1008', '1892', '2070', '2988', '3059'] | 2024-04-10 21:00:00 | 2024-04-10 21:00:00 | ['243363'] |
| 10 | ['1008', '1892', '2070', '3059', '4543'] | 2024-04-11 16:00:00 | 2024-07-28 13:50:00 | ['243076', '242862', '242866', '242868', '243089', '243093', '243755', '243736', '244143', '244146', '244149', '244223', '244226', '255402', '244366', '244604', '244377', '244607', '244612', '245017', '245045', '245053', '245061', '245064', '245072', '245375', '245387', '245392', '248202', '248205', '245395', '248968', '248975', '248985', '248991', '248936', '249001', '249003', '249013', '249198', '249201', '249218', '251280', '251298', '251311', '251313', '251323', '250774'] |
| 11 | ['1008', '1343', '1892', '2070', '3059'] | 2024-08-25 17:45:00 | 2024-08-26 14:20:00 | ['253915', '253918'] |
| 12 | ['1008', '1892', '2070', '3059', '7784'] | 2024-08-29 16:30:00 | 2024-08-29 16:30:00 | ['253313'] |
| 13 | ['1008', '1343', '1892', '2070', '3059'] | 2024-08-29 20:30:00 | 2024-08-29 20:30:00 | ['253319'] |
| 14 | ['1008', '1892', '2070', '3059', '7784'] | 2024-08-30 16:30:00 | 2024-08-30 20:30:00 | ['254021', '254026'] |
| 15 | ['1008', '1343', '1892', '2070', '3059'] | 2024-09-24 20:00:00 | 2024-11-23 07:00:00 | ['255592', '255786', '255597', '255599', '255837', '256158', '256522', '256732', '256735', '256839', '256906', '256923', '256926', '257171', '257186', '257071', '259746', '259758', '259816'] |
| 16 | ['1343', '1892', '1954', '2070', '7784'] | 2025-01-21 12:00:00 | 2025-04-02 15:00:00 | ['262924', '263040', '263243', '263440', '263141', '263460', '263151', '264267', '263156', '263160', '266486', '266660', '266673', '266679', '267868', '267925'] |
| 17 | ['156', '157', '197', '4903', '6237'] | 2025-04-06 14:30:00 | 2025-06-14 20:25:00 | ['267878', '268218', '268254', '268316', '268428', '269672', '270110', '270127', '270135', '270145', '270411', '270672', '270678', '270070', '271030', '271035', '271037', '270701', '271264', '271272', '271300', '271312'] |
| 18 | ['157', '197', '2776', '4903', '6237'] | 2025-07-26 19:30:00 | 2025-07-29 14:30:00 | ['273614', '274134', '274270'] |
| 19 | ['156', '157', '197', '4903', '6237'] | 2025-08-08 15:05:00 | 2025-12-01 19:20:00 | ['274554', '275422', '276047', '275411', '276494', '276499', '276281', '277228', '277352', '277366', '277870', '279266', '279282', '279437', '279491', '279520', '279612', '279615', '279618', '279621', '279626', '280514', '280608', '280646', '280676', '280706', '280787', '280794', '280800', '282222', '282410', '282439', '282466'] |
| 20 | ['156', '157', '197', '4896', '4903'] | 2026-01-13 22:20:00 | 2026-06-20 16:45:00 | ['283616', '284263', '284560', '285728', '285752', '284614', '286019', '286031', '286286', '286346', '286974', '286985', '287077', '287106', '287135', '287808', '288043', '288083', '288167', '288319', '288325', '288328', '288442', '288459', '288460', '288892', '288897', '289688', '289700', '289708', '289938', '290416', '290424', '290428', '292454', '292706', '292815', '292917', '293109', '293113', '294490', '294809', '294876', '294922', '294996', '295003'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['1343', '1892', '1954', '2070', '7784'] (active 2025-01-21 12:00:00 to 2025-04-02 15:00:00) vs roster ['156', '157', '197', '4903', '6237'] (active 2025-04-06 14:30:00 to 2025-06-14 20:25:00) (3-day gap)

**Assessment**: (a) same organization with roster turnover
**Proposed decision**: `KEEP_AS_SINGLE_TEAM`
**Rationale**: 267 matches under this name with clean sequential roster eras (each era's appearances are chronologically disjoint from the next - a roster hand-off pattern, not simultaneous usage). Per the review's own rule, zero overlap between two eras of an established, high-volume name is NOT by itself evidence of a different organization - full roster turnover over a multi-year dataset is expected for real orgs. Recommend keeping as a single identity, with the era boundaries available for anyone who later wants roster-level (not org-level) granularity.

## `BIG EQUIPA`

- First appearance: 2024-05-31 19:25:00  |  Last appearance: 2026-04-01 16:00:00
- Matches: 19  |  Tournaments (6): ['CCT Season 3 European Series #15', 'ESL Challenger League Season 51: Europe — Cup #3', 'ESL Impact League Season 5', 'ESL Impact League Season 6', 'ESL Impact League Season 6: European Division', 'ESL Impact League Season 8']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['5669', '679'] | 2024-05-31 19:25:00 | 2024-10-19 17:30:00 | ['248246', '248249', '248252', '254322', '254483', '254490', '254496', '254500', '254312'] |
| 2 | ['679'] | 2024-11-22 15:00:00 | 2024-11-24 12:00:00 | ['259864', '259867', '259871', '259874'] |
| 3 | ['26110', '4969', '5591', '5669', '8136'] | 2025-11-30 15:00:00 | 2025-11-30 17:15:00 | ['282478', '282481'] |
| 4 | ['26110', '4969', '8136'] | 2026-02-09 22:20:00 | 2026-02-12 15:00:00 | ['286494', '286698', '286880'] |
| 5 | ['5669', '679'] | 2026-04-01 16:00:00 | 2026-04-01 16:00:00 | ['292149'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['26110', '4969', '8136'] (active 2026-02-09 22:20:00 to 2026-02-12 15:00:00) vs roster ['5669', '679'] (active 2026-04-01 16:00:00 to 2026-04-01 16:00:00) (48-day gap)

**Assessment**: (c) insufficient evidence
**Proposed decision**: `MANUAL_REVIEW`
**Rationale**: Only 19 matches under this name; roster eras are sequential (consistent with turnover) but the sample is too small to be confident this isn't actually two different low-tier squads that happened to reuse a common-sounding name. Needs manual review.

## `BRUTE`

- First appearance: 2026-01-24 18:00:00  |  Last appearance: 2026-02-19 12:00:00
- Matches: 7  |  Tournaments (2): ['CCT Season 3 European Series #16', 'NODWIN Clutch Series #4']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['1674'] | 2026-01-24 18:00:00 | 2026-01-29 12:00:00 | ['285217', '285537', '285717', '285781'] |
| 2 | ['25794', '4493'] | 2026-02-16 15:00:00 | 2026-02-19 12:00:00 | ['287013', '287117', '287197'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['1674'] (active 2026-01-24 18:00:00 to 2026-01-29 12:00:00) vs roster ['25794', '4493'] (active 2026-02-16 15:00:00 to 2026-02-19 12:00:00) (18-day gap)

**Assessment**: (c) insufficient evidence
**Proposed decision**: `MANUAL_REVIEW`
**Rationale**: Only 7 matches under this name; roster eras are sequential (consistent with turnover) but the sample is too small to be confident this isn't actually two different low-tier squads that happened to reuse a common-sounding name. Needs manual review.

## `ECSTATIC`

- First appearance: 2023-10-24 12:00:00  |  Last appearance: 2026-04-24 20:00:00
- Matches: 183  |  Tournaments (53): ['A1 Gaming League Season 9', 'BLAST Bounty Fall 2025. Closed Qualifiers', 'BLAST Bounty Winter 2026. Closed Qualifiers', 'BLAST Open London 2025. Closed Qualifiers', 'CCT Central Europe Series #8', 'CCT East Europe Series #3', 'CCT Season 2 European Series #11', 'CCT Season 2 European Series #12', 'CCT Season 2 European Series #13', 'CCT Season 2 European Series #14', 'CCT Season 2 European Series #15', 'CCT Season 2 European Series #16', 'CCT Season 2 European Series #17', 'CCT Season 2 European Series #18', 'CCT Season 2 European Series #20', 'CCT Season 2 European Series #3', 'CCT Season 2 European Series #9', 'CCT Season 3 European Series #1', 'CCT Season 3 European Series #10', 'CCT Season 3 European Series #12', 'CCT Season 3 European Series #13', 'CCT Season 3 European Series #14', 'CCT Season 3 European Series #15', 'CCT Season 3 European Series #8', 'CCT Season 3 European Series #9', 'CS2 Asia Championships 2025. Qualifier', 'ESL Challenger Atlanta 2024. Qualifier Europe', 'ESL Challenger Jonköping 2024. Open Qualifier', 'European Pro League Season 20', 'Exort The Proving Grounds Season 3', 'Exort The Proving Grounds Season 5', 'Galaxy Battle 2025 // Phase 5', 'IEM Dallas 2024: Qualifier Europe', 'NODWIN Clutch Series #4', 'NODWIN Clutch Series #5', 'NODWIN Clutch Series #7', 'Nordic Masters Fall 2024', 'PGL Astana 2025. Closed European Qualifier', 'PGL Astana 2025. Open Qualifier', 'PGL CS2 Major Copenhagen 2024: Closed Qualifiers', 'PGL CS2 Major Copenhagen 2024: European RMR B', 'PGL CS2 Major Copenhagen 2024: Open Qualifiers', 'PGL Major Copenhagen 2024', 'Parken Challenger Championship #1', 'RES European Series #1', 'RES Showdown Fall 2025', 'StarLadder StarSeries Fall 2025. Closed Qualifier', 'Thunderpick World Championship 2024 - Qualifiers Europe #1', 'Thunderpick World Championship 2024 - Qualifiers Europe #2', 'Thunderpick World Championship 2025. Closed Qualifier', 'Thunderpick World Championship 2025: European Series #1', 'YaLLa Compass Fall 2023', 'YaLLa Compass Winter 2025']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['1676', '1677', '3241', '5128', '708'] | 2023-10-24 12:00:00 | 2023-10-24 12:00:00 | ['233424'] |
| 2 | ['1676', '3241', '3457', '5128', '708'] | 2023-10-26 21:00:00 | 2024-04-04 12:30:00 | ['233610', '233705', '234075', '234077', '234300', '234890', '235159', '235339', '235477', '236314', '236334', '236821', '236860', '236865', '237440', '237709', '237724', '237875', '237935', '237975', '239312', '239730', '239735', '239764', '239809', '240274', '240276', '240279', '240417', '240464', '240470', '240478', '240540', '241639', '241667', '241753', '241912', '241932', '241997', '242051', '242525', '242583', '242777'] |
| 3 | ['1681', '2538', '3384', '3933', '7245'] | 2024-05-23 20:00:00 | 2024-05-23 20:00:00 | ['245117'] |
| 4 | ['1681', '3246', '3384', '3933', '7245'] | 2024-08-05 14:00:00 | 2024-09-23 13:30:00 | ['252221', '252226', '252229', '252242', '252847', '253253', '253325', '253370', '253363', '253411', '253959', '253997', '254092', '254657', '254544', '254703', '254877', '255203', '255217', '255249', '255366', '255395', '255507', '255602'] |
| 5 | ['1091', '1681', '3384', '3933', '7245'] | 2024-09-23 21:00:00 | 2024-09-23 21:00:00 | ['255684'] |
| 6 | ['1681', '3246', '3384', '3933', '7245'] | 2024-09-25 17:30:00 | 2024-09-25 17:30:00 | ['255785'] |
| 7 | ['1091', '1681', '3384', '3933', '7245'] | 2024-09-26 21:00:00 | 2024-09-26 21:00:00 | ['255834'] |
| 8 | ['1681', '3246', '3384', '3933', '7245'] | 2024-09-27 17:30:00 | 2024-09-27 17:30:00 | ['255836'] |
| 9 | ['1091', '1681', '3384', '3933', '7245'] | 2024-09-28 21:00:00 | 2024-12-15 19:30:00 | ['256138', '256201', '256239', '256317', '256448', '256525', '256571', '256582', '256621', '256687', '258159', '258162', '258167', '258193', '260104', '260310', '260346', '260408', '260514', '260715', '260716', '260782', '260785'] |
| 10 | ['1091', '1681', '3384', '6665', '7245'] | 2025-01-15 12:00:00 | 2025-08-29 13:00:00 | ['262247', '262932', '263242', '263448', '263527', '263561', '263581', '263805', '263903', '264803', '264884', '265015', '265081', '265258', '265843', '266141', '266375', '266607', '266654', '266670', '266678', '266682', '267235', '267290', '269823', '269841', '269958', '270018', '269954', '270086', '270160', '270383', '270633', '270649', '274538', '274543', '274545', '274573', '275684', '275693', '276006', '276011', '276013', '276215', '276406', '276428', '276510', '276529'] |
| 11 | ['1091', '3384', '641', '6665', '7245'] | 2025-09-09 11:30:00 | 2025-12-11 15:00:00 | ['277506', '279586', '279854', '279856', '279956', '280708', '280804', '280824', '281184', '282796', '282807', '282870', '282885', '283230'] |
| 12 | ['1091', '2152', '3384', '4600', '7245'] | 2026-01-15 22:30:00 | 2026-04-24 20:00:00 | ['283613', '285666', '286274', '286052', '286260', '286298', '286343', '286283', '287096', '287132', '287150', '287277', '287284', '287287', '287295', '287298', '287333', '287343', '287352', '287355', '287664', '287663', '287681', '287682', '291538'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['1676', '3241', '3457', '5128', '708'] (active 2023-10-26 21:00:00 to 2024-04-04 12:30:00) vs roster ['1681', '2538', '3384', '3933', '7245'] (active 2024-05-23 20:00:00 to 2024-05-23 20:00:00) (49-day gap)

**Assessment**: (a) same organization with roster turnover
**Proposed decision**: `KEEP_AS_SINGLE_TEAM`
**Rationale**: 183 matches under this name with clean sequential roster eras (each era's appearances are chronologically disjoint from the next - a roster hand-off pattern, not simultaneous usage). Per the review's own rule, zero overlap between two eras of an established, high-volume name is NOT by itself evidence of a different organization - full roster turnover over a multi-year dataset is expected for real orgs. Recommend keeping as a single identity, with the era boundaries available for anyone who later wants roster-level (not org-level) granularity.

## `ENTERPRISE esports`

- First appearance: 2024-01-15 22:30:00  |  Last appearance: 2024-09-03 12:00:00
- Matches: 29  |  Tournaments (11): ['CCT Season 2 European Series #2', 'ESL Challenger Atlanta 2024. Qualifier Europe', 'Elisa Invitational Fall 2024', 'Elisa Invitational Spring 2024', 'PGL CS2 Major Copenhagen 2024: European RMR A', 'PGL CS2 Major Copenhagen 2024: Open Qualifiers', 'Perfect World Shanghai Major 2024: EU Qualifier', 'RES European Series #2', 'Regional Clash Arena Europe', 'Thunderpick World Championship 2024 - Qualifiers Europe #1', 'YaLLa Compass Spring 2024']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['2243', '2840', '3045', '3271'] | 2024-01-15 22:30:00 | 2024-01-16 20:00:00 | ['236962', '236967'] |
| 2 | ['2362', '310', '528', '5507'] | 2024-02-14 16:15:00 | 2024-05-10 15:00:00 | ['239196', '239534', '239550', '239569', '242731', '242771', '243095', '243345', '243489', '242894', '243509', '243930', '243948', '244019', '244229', '255425'] |
| 3 | ['2362', '528', '5507', '716'] | 2024-06-13 15:00:00 | 2024-06-16 13:00:00 | ['249097', '249104'] |
| 4 | ['2362', '310', '528', '5507'] | 2024-08-15 15:00:00 | 2024-09-03 12:00:00 | ['252873', '253323', '253369', '253418', '253061', '253647', '253685', '253763', '254376'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['2243', '2840', '3045', '3271'] (active 2024-01-15 22:30:00 to 2024-01-16 20:00:00) vs roster ['2362', '310', '528', '5507'] (active 2024-02-14 16:15:00 to 2024-05-10 15:00:00) (28-day gap)

**Assessment**: (c) insufficient evidence
**Proposed decision**: `MANUAL_REVIEW`
**Rationale**: Only 29 matches under this name; roster eras are sequential (consistent with turnover) but the sample is too small to be confident this isn't actually two different low-tier squads that happened to reuse a common-sounding name. Needs manual review.

## `Entropiq`

- First appearance: 2023-11-16 18:00:00  |  Last appearance: 2024-03-05 14:00:00
- Matches: 15  |  Tournaments (4): ['PGL CS2 Major Copenhagen 2024: Closed Qualifiers', 'PGL CS2 Major Copenhagen 2024: Open Qualifiers', 'YaLLa Compass Fall 2023', 'YaLLa Compass Spring 2024']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['1328', '1487', '165', '2787', '2838'] | 2023-11-16 18:00:00 | 2023-12-04 16:45:00 | ['234314', '234878', '235147', '235330'] |
| 2 | ['1644', '4379', '5198', '5271', '700'] | 2024-01-09 21:15:00 | 2024-01-10 02:10:00 | ['236329', '236337', '236343'] |
| 3 | ['1328', '1487', '165', '2787', '4236'] | 2024-01-18 19:10:00 | 2024-03-05 14:00:00 | ['237464', '237692', '237744', '237886', '238816', '239050', '239336', '240532'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['1328', '1487', '165', '2787', '2838'] (active 2023-11-16 18:00:00 to 2023-12-04 16:45:00) vs roster ['1644', '4379', '5198', '5271', '700'] (active 2024-01-09 21:15:00 to 2024-01-10 02:10:00) (36-day gap)
- roster ['1644', '4379', '5198', '5271', '700'] (active 2024-01-09 21:15:00 to 2024-01-10 02:10:00) vs roster ['1328', '1487', '165', '2787', '4236'] (active 2024-01-18 19:10:00 to 2024-03-05 14:00:00) (8-day gap)

**Assessment**: (c) insufficient evidence
**Proposed decision**: `MANUAL_REVIEW`
**Rationale**: Only 15 matches under this name; roster eras are sequential (consistent with turnover) but the sample is too small to be confident this isn't actually two different low-tier squads that happened to reuse a common-sounding name. Needs manual review.

## `For The Win eSports`

- First appearance: 2024-01-16 23:15:00  |  Last appearance: 2024-01-18 01:30:00
- Matches: 3  |  Tournaments (2): ['IEM Chengdu 2024. Open Qualifier', 'PGL CS2 Major Copenhagen 2024: Open Qualifiers']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['3114'] | 2024-01-16 23:15:00 | 2024-01-16 23:15:00 | ['237408'] |
| 2 | ['1365', '2716', '3513', '7395'] | 2024-01-17 23:45:00 | 2024-01-18 01:30:00 | ['237477', '237481'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['3114'] (active 2024-01-16 23:15:00 to 2024-01-16 23:15:00) vs roster ['1365', '2716', '3513', '7395'] (active 2024-01-17 23:45:00 to 2024-01-18 01:30:00) (1-day gap)

**Assessment**: (c) insufficient evidence
**Proposed decision**: `MANUAL_REVIEW`
**Rationale**: Only 3 matches under this name; roster eras are sequential (consistent with turnover) but the sample is too small to be confident this isn't actually two different low-tier squads that happened to reuse a common-sounding name. Needs manual review.

## `Heroic`

- First appearance: 2023-10-25 16:00:00  |  Last appearance: 2026-06-04 20:00:00
- Matches: 235  |  Tournaments (63): ['BC.Game Masters Championship', 'BLAST Bounty Fall 2025. Closed Qualifiers', 'BLAST Bounty Spring 2025', 'BLAST Bounty Spring 2025. Closed Qualifiers', 'BLAST Bounty Winter 2026', 'BLAST Bounty Winter 2026. Closed Qualifiers', 'BLAST Premier: Fall Final 2023', 'BLAST Premier: Fall Groups 2024', 'BLAST Premier: Fall Showdown 2024', 'BLAST Premier: Spring Groups 2024', 'BLAST Premier: Spring Showdown 2024', 'BLAST Premier: World Final 2023', 'BLAST Rising Europe Spring 2025', 'BLAST.tv Austin Major 2025', 'BLAST.tv Austin Major 2025. Closed Qualifier', 'BetBoom Dacha Belgrade 2024 #2', 'BetBoom Dacha CS2 Belgrade 2024', 'CCT Season 2 Global Finals', 'CCT Season 3 European Series #11', 'CS2 Asia Championships 2025', 'CS2 Asia Championships 2025. Qualifier', 'ESL Pro League Season 19', 'ESL Pro League Season 20', 'ESL Pro League Season 21', 'ESL Pro League Season 22', 'ESL Pro League Season 23', 'Elisa Masters Espoo 2024', 'Esports World Cup 2024 по CS2. Open Qualifier', 'Esports World Cup 2025 по CS2', 'FISSURE PLAYGROUND 1 — CS', 'FISSURE PLAYGROUND 2 — CS', 'IEM Chengdu 2024', 'IEM Chengdu 2025', 'IEM Cologne 2024', 'IEM Cologne 2025', 'IEM Cologne Major 2026', 'IEM Dallas 2024', 'IEM Dallas 2024: Qualifier Europe', 'IEM Dallas 2025', 'IEM Dallas 2025: Qualifier Europe', 'IEM Katowice 2024', 'IEM Katowice 2025', 'IEM Krakow 2026', 'IEM Rio 2024', 'IEM Rio 2024: Qualifier Europe', 'MESA Nomadic Masters: Spring 2025', 'PGL Astana 2026', 'PGL Bucharest 2025. European Qualifiers', 'PGL CS2 Major Copenhagen 2024: Closed Qualifiers', 'PGL CS2 Major Copenhagen 2024: European RMR B', 'PGL CS2 Major Copenhagen 2024: Open Qualifiers', 'PGL Cluj-Napoca 2026', 'PGL Major Copenhagen 2024', 'PGL Masters Bucharest 2025', 'Perfect World Shanghai Major 2024', 'Perfect World Shanghai Major 2024: European RMR B', 'Roman Imperium Cup VII', 'Roobet Cup 2023', 'Stake Ranked Episode 1', 'Stake Ranked Episode 2', 'Thunderpick World Championship 2023', 'Thunderpick World Championship 2024', 'YaLLa Compass 2024']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['12', '6470', '6665', '709', '945'] | 2023-10-25 16:00:00 | 2023-11-04 15:00:00 | ['233371', '233375', '233513', '233651', '233657', '233861'] |
| 2 | ['12', '2848', '439', '709', '945'] | 2023-11-22 17:50:00 | 2023-11-24 17:30:00 | ['234118', '234125', '235080'] |
| 3 | ['12', '2848', '5128', '709', '945'] | 2023-12-13 13:35:00 | 2023-12-14 14:30:00 | ['235300', '235303'] |
| 4 | ['1548', '2152', '2611', '709', '945'] | 2024-01-09 18:00:00 | 2024-05-04 18:05:00 | ['236251', '236438', '236453', '236459', '236565', '236593', '236597', '237468', '237684', '237732', '237882', '237942', '237068', '237072', '237903', '238869', '237926', '239002', '239012', '239015', '239320', '239734', '239740', '239772', '239811', '240217', '240231', '241001', '241151', '241162', '241165', '240547', '241645', '241669', '241906', '241926', '242003', '242049', '242483', '243118', '243122', '243125', '243753', '243217', '243720', '243724', '243726', '243730'] |
| 5 | ['1002', '1548', '2611', '709', '945'] | 2024-05-15 16:00:00 | 2024-05-18 18:00:00 | ['244367', '244376', '244379', '244876'] |
| 6 | ['1548', '2152', '2611', '709', '945'] | 2024-05-28 00:20:00 | 2024-05-31 21:00:00 | ['244892', '245544', '245548', '245551', '245840'] |
| 7 | ['1002', '1548', '2611', '709', '945'] | 2024-06-05 11:10:00 | 2024-06-06 18:30:00 | ['245744', '245754', '245772', '245778', '245783'] |
| 8 | ['1548', '2611', '709', '945', '981'] | 2024-07-29 14:50:00 | 2024-08-02 13:00:00 | ['250803', '251675', '251678'] |
| 9 | ['1548', '2611', '3168', '709', '945'] | 2024-08-07 15:15:00 | 2024-08-07 15:15:00 | ['251446'] |
| 10 | ['1002', '1548', '2611', '709', '945'] | 2024-08-08 17:45:00 | 2024-12-13 09:00:00 | ['251970', '251979', '252988', '253859', '253863', '253873', '253875', '253877', '253928', '253941', '249794', '254385', '254397', '254404', '254407', '255350', '255335', '256314', '256643', '256645', '256662', '256952', '256962', '256971', '256973', '257166', '257184', '257574', '257605', '257608', '257074', '259751', '259761', '259808', '260136', '260277', '260293', '260315', '260336', '260389'] |
| 11 | ['1451', '1661', '2776', '3395', '5364'] | 2025-01-14 16:30:00 | 2025-01-18 14:00:00 | ['261933', '262781'] |
| 12 | ['1451', '1661', '2776', '5364', '982'] | 2025-01-23 18:00:00 | 2025-01-23 18:00:00 | ['262943'] |
| 13 | ['1451', '1661', '2776', '3395', '5364'] | 2025-01-29 20:55:00 | 2025-06-10 18:00:00 | ['262735', '263546', '263558', '264277', '264275', '264383', '264497', '264510', '264517', '264520', '265566', '265700', '265806', '265912', '266052', '266058', '266061', '266062', '266132', '268276', '268702', '268758', '269090', '269095', '269104', '269108', '269111', '269099', '269356', '269362', '269376', '266017', '270071', '271024', '271032', '271036', '269926', '271140', '271154', '271191', '271216', '271233', '271243', '271253'] |
| 14 | ['1451', '3368', '3395', '5364', '7784'] | 2025-07-15 21:35:00 | 2025-08-10 17:35:00 | ['273566', '273569', '273593', '274071', '274113', '273617', '274137', '274564', '275423'] |
| 15 | ['1451', '26127', '3368', '3395', '5364'] | 2025-08-11 17:00:00 | 2025-08-22 17:30:00 | ['275688', '275413', '276495'] |
| 16 | ['1451', '26127', '2776', '3368', '5364'] | 2025-09-13 11:00:00 | 2025-10-01 17:00:00 | ['277230', '277362', '277377', '278562', '278917', '279008', '279195'] |
| 17 | ['1451', '2776', '3368', '5364', '948'] | 2025-10-14 10:55:00 | 2025-10-19 06:00:00 | ['279882', '279888', '279891', '279926', '279931'] |
| 18 | ['1451', '26127', '2776', '3368', '5364'] | 2025-10-26 14:00:00 | 2025-11-23 17:10:00 | ['280506', '280606', '280647', '280789', '280553', '280946', '280951', '281932', '281940', '281959'] |
| 19 | ['2776', '3368', '4042', '5364', '6470'] | 2026-01-14 17:00:00 | 2026-06-04 20:00:00 | ['283626', '284259', '284737', '284562', '285731', '286351', '286979', '286995', '287081', '287476', '287548', '287572', '287648', '287745', '287803', '288053', '288093', '289026', '289088', '289092', '289100', '289104', '289154', '289268', '289279', '289283', '289288', '292455', '292709', '292816', '292918', '293784', '293790', '292758', '293988', '294019', '294113'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['1002', '1548', '2611', '709', '945'] (active 2024-08-08 17:45:00 to 2024-12-13 09:00:00) vs roster ['1451', '1661', '2776', '3395', '5364'] (active 2025-01-14 16:30:00 to 2025-01-18 14:00:00) (32-day gap)
- roster ['1002', '1548', '2611', '709', '945'] (active 2024-08-08 17:45:00 to 2024-12-13 09:00:00) vs roster ['1451', '1661', '2776', '5364', '982'] (active 2025-01-23 18:00:00 to 2025-01-23 18:00:00) (41-day gap)
- roster ['1002', '1548', '2611', '709', '945'] (active 2024-08-08 17:45:00 to 2024-12-13 09:00:00) vs roster ['1451', '1661', '2776', '3395', '5364'] (active 2025-01-29 20:55:00 to 2025-06-10 18:00:00) (47-day gap)

**Assessment**: (a) same organization with roster turnover
**Proposed decision**: `KEEP_AS_SINGLE_TEAM`
**Rationale**: 235 matches under this name with clean sequential roster eras (each era's appearances are chronologically disjoint from the next - a roster hand-off pattern, not simultaneous usage). Per the review's own rule, zero overlap between two eras of an established, high-volume name is NOT by itself evidence of a different organization - full roster turnover over a multi-year dataset is expected for real orgs. Recommend keeping as a single identity, with the era boundaries available for anyone who later wants roster-level (not org-level) granularity.

## `MANA eSports`

- First appearance: 2025-11-22 18:20:00  |  Last appearance: 2026-04-06 20:00:00
- Matches: 10  |  Tournaments (4): ['ESL Challenger League Season 51: Europe — Cup #3', 'European Pro League Series 3', 'IstanbuLAN 2026', 'Roman Imperium Cup IV']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['1354'] | 2025-11-22 18:20:00 | 2025-11-26 12:00:00 | ['282185', '282187'] |
| 2 | ['1353'] | 2026-01-17 20:00:00 | 2026-04-06 20:00:00 | ['284620', '286861', '286870', '292162', '292201', '292218', '292225', '292231'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['1354'] (active 2025-11-22 18:20:00 to 2025-11-26 12:00:00) vs roster ['1353'] (active 2026-01-17 20:00:00 to 2026-04-06 20:00:00) (52-day gap)

**Assessment**: (c) insufficient evidence
**Proposed decision**: `MANUAL_REVIEW`
**Rationale**: Only 10 matches under this name; roster eras are sequential (consistent with turnover) but the sample is too small to be confident this isn't actually two different low-tier squads that happened to reuse a common-sounding name. Needs manual review.

## `MOUZ NXT`

- First appearance: 2023-10-25 21:00:00  |  Last appearance: 2026-05-11 14:00:00
- Matches: 106  |  Tournaments (34): ['BC.Game Masters Championship', 'CCT East Europe Series #3', 'CCT Europe 2026 Series #1', 'CCT Online Finals #4', 'CCT Season 2 European Series #1', 'CCT Season 2 European Series #2', 'CCT Season 2 European Series #3', 'CCT Season 2 European Series #4', 'CCT Season 2 European Series #5', 'CCT Season 2 European Series #6', 'CCT Season 2 European Series #7', 'CCT Season 2 European Series #8', 'CCT Season 2 European Series #9', 'CCT Season 3 European Series #15', 'CCT Season 3 European Series #16', 'CCT Season 3 European Series #17', 'ESL Challenger Atlanta 2024. Qualifier Europe', 'Elisa Invitational Fall 2024', 'Elisa Invitational Spring 2024', 'European Pro League Series 3', 'European Pro League Series 4', 'IEM Atlanta 2026. Open Qualifier', 'NODWIN Clutch Series #4', 'PGL Bucharest 2026. European Qualifier', 'RES European Series #2', 'RES European Series #3', 'RES European Series #4', 'RES European Series #5', 'RES European Series #6', 'Red Bull GIBAWAY JOURNEY 2025', 'Roman Imperium Cup IV', 'Thunderpick World Championship 2024 - Qualifiers Europe #1', 'Thunderpick World Championship 2024 - Qualifiers Europe #2', 'Urban Riga Open #2']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['4010', '4029', '6284', '6470', '6665'] | 2023-10-25 21:00:00 | 2023-11-14 13:30:00 | ['233428', '234256', '234258'] |
| 2 | ['4010', '6284', '6470', '6665'] | 2024-02-26 16:00:00 | 2024-03-10 13:00:00 | ['243272', '243285', '243299'] |
| 3 | ['4010', '5309', '6284', '6470', '6665'] | 2024-04-03 21:30:00 | 2024-04-03 21:30:00 | ['242804'] |
| 4 | ['4010', '6284', '6470', '6665'] | 2024-04-08 15:00:00 | 2024-04-08 15:00:00 | ['243336'] |
| 5 | ['4010', '5309', '6284', '6470', '6665'] | 2024-04-15 12:00:00 | 2024-07-27 18:10:00 | ['243480', '242792', '243929', '243950', '243951', '243982', '244009', '244861', '244338', '244868', '244871', '255403', '244659', '244820', '244823', '244827', '245141', '245149', '245126', '245279', '245274', '245323', '245416', '245409', '245557', '248229', '248231', '248275', '248304', '248334', '248440', '248340', '248563', '248950', '249233', '251165', '251163', '251508', '251507'] |
| 6 | ['1447', '4010', '5309', '6284', '6665'] | 2024-08-03 21:30:00 | 2024-08-03 21:30:00 | ['252206'] |
| 7 | ['1091', '4010', '5309', '6284', '6665'] | 2024-08-14 12:00:00 | 2024-08-15 15:00:00 | ['252866', '253188'] |
| 8 | ['1447', '4010', '5309', '6284', '6665'] | 2024-08-16 23:20:00 | 2024-08-16 23:20:00 | ['253339'] |
| 9 | ['4010', '5309', '6284', '6665'] | 2024-08-17 19:00:00 | 2024-08-17 20:05:00 | ['253377', '253386'] |
| 10 | ['1091', '4010', '5309', '6284', '6665'] | 2024-08-21 12:00:00 | 2024-08-21 12:00:00 | ['253424'] |
| 11 | ['2126', '2200', '2726', '3121', '315'] | 2024-08-26 21:30:00 | 2024-08-27 12:30:00 | ['253956', '254091'] |
| 12 | ['1447', '4010', '5309', '6284', '6665'] | 2024-08-28 18:30:00 | 2024-08-28 18:30:00 | ['253983'] |
| 13 | ['1091', '4010', '5309', '6284', '6665'] | 2024-08-29 15:00:00 | 2024-09-10 13:00:00 | ['253752', '254908'] |
| 14 | ['26888', '26931', '27860', '5165'] | 2025-11-07 14:00:00 | 2025-12-06 12:00:00 | ['281264', '281271', '281298', '281323', '281352', '282213', '282218', '282220', '282527', '282542'] |
| 15 | ['25669', '26888', '26931', '27860', '5165'] | 2026-01-05 11:00:00 | 2026-03-25 19:40:00 | ['283670', '283675', '283678', '284616', '284624', '285204', '285530', '285710', '285803', '285791', '285994', '286045', '286466', '286477', '287012', '287094', '287126', '287133', '287208', '287407', '287423', '287457', '287433', '287530', '287636', '287714', '288104', '288213', '288309', '289035'] |
| 16 | ['25669', '27860', '5165', '7774'] | 2026-05-04 17:00:00 | 2026-05-11 14:00:00 | ['292484', '292528', '292606', '292692', '292862'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['4010', '5309', '6284', '6470', '6665'] (active 2024-04-15 12:00:00 to 2024-07-27 18:10:00) vs roster ['2126', '2200', '2726', '3121', '315'] (active 2024-08-26 21:30:00 to 2024-08-27 12:30:00) (30-day gap)
- roster ['1447', '4010', '5309', '6284', '6665'] (active 2024-08-03 21:30:00 to 2024-08-03 21:30:00) vs roster ['2126', '2200', '2726', '3121', '315'] (active 2024-08-26 21:30:00 to 2024-08-27 12:30:00) (23-day gap)
- roster ['1091', '4010', '5309', '6284', '6665'] (active 2024-08-14 12:00:00 to 2024-08-15 15:00:00) vs roster ['2126', '2200', '2726', '3121', '315'] (active 2024-08-26 21:30:00 to 2024-08-27 12:30:00) (11-day gap)
- roster ['1447', '4010', '5309', '6284', '6665'] (active 2024-08-16 23:20:00 to 2024-08-16 23:20:00) vs roster ['2126', '2200', '2726', '3121', '315'] (active 2024-08-26 21:30:00 to 2024-08-27 12:30:00) (9-day gap)
- roster ['4010', '5309', '6284', '6665'] (active 2024-08-17 19:00:00 to 2024-08-17 20:05:00) vs roster ['2126', '2200', '2726', '3121', '315'] (active 2024-08-26 21:30:00 to 2024-08-27 12:30:00) (9-day gap)
- roster ['1091', '4010', '5309', '6284', '6665'] (active 2024-08-21 12:00:00 to 2024-08-21 12:00:00) vs roster ['2126', '2200', '2726', '3121', '315'] (active 2024-08-26 21:30:00 to 2024-08-27 12:30:00) (5-day gap)
- roster ['2126', '2200', '2726', '3121', '315'] (active 2024-08-26 21:30:00 to 2024-08-27 12:30:00) vs roster ['1447', '4010', '5309', '6284', '6665'] (active 2024-08-28 18:30:00 to 2024-08-28 18:30:00) (1-day gap)
- roster ['2126', '2200', '2726', '3121', '315'] (active 2024-08-26 21:30:00 to 2024-08-27 12:30:00) vs roster ['1091', '4010', '5309', '6284', '6665'] (active 2024-08-29 15:00:00 to 2024-09-10 13:00:00) (2-day gap)

**Assessment**: (a) same organization with roster turnover
**Proposed decision**: `KEEP_AS_SINGLE_TEAM`
**Rationale**: 106 matches under this name with clean sequential roster eras (each era's appearances are chronologically disjoint from the next - a roster hand-off pattern, not simultaneous usage). Per the review's own rule, zero overlap between two eras of an established, high-volume name is NOT by itself evidence of a different organization - full roster turnover over a multi-year dataset is expected for real orgs. Recommend keeping as a single identity, with the era boundaries available for anyone who later wants roster-level (not org-level) granularity.

## `Ninjas in Pyjamas`

- First appearance: 2023-10-26 19:00:00  |  Last appearance: 2026-05-30 19:30:00
- Matches: 182  |  Tournaments (56): ['BLAST Bounty Fall 2025. Closed Qualifiers', 'BLAST Bounty Winter 2026. Closed Qualifiers', 'BLAST Open Spring 2026', 'BLAST Premier: Fall Final 2023', 'BLAST Premier: Fall Groups 2024', 'BLAST Premier: Fall Showdown 2024', 'BLAST Premier: Spring Groups 2024', 'BLAST Premier: Spring Showdown 2024', 'BLAST Rising Europe Spring 2025. Closed Qualifier', 'CCT Season 2 European Series #13', 'CCT Season 3 European Series #2', 'CCT Season 3 European Series #3', 'CS Asia Championships 2026', 'CS2 Asia Championships 2023', 'CS2 Asia Championships 2025. Qualifier', 'Copenhagen Gaming Week 2024', 'DraculaN #1', 'ESL Challenger Atlanta 2024. Qualifier Europe', 'ESL Challenger Jonköping 2024. Closed Qualifier', 'ESL Challenger Jonköping 2024. Open Qualifier', 'ESL Pro League Season 19', 'ESL Pro League Season 20', 'ESL Pro League Season 23', 'Elisa Masters Espoo 2024', 'Esports World Cup 2024 по CS2. Open Qualifier', 'FiReLEAGUE Global Finals 2024', 'IEM Chengdu 2024. Open Qualifier', 'IEM Cologne 2025', 'IEM Krakow 2026', 'IEM Rio 2024: Qualifier Europe', 'PGL Astana 2025', 'PGL Astana 2025. Closed European Qualifier', 'PGL Astana 2025. Open Qualifier', 'PGL Bucharest 2025. European Qualifiers', 'PGL CS2 Major Copenhagen 2024: European RMR A', 'PGL Masters Bucharest 2025', 'Perfect World Shanghai Major 2024: European RMR B', 'RES European Series #4', 'RES European Series #6', 'RES Showdown Spring 2026', 'Roman Imperium Cup V', 'Roman Imperium Cup VI', 'Roobet Cup 2023', 'Skyesports Masters 2024', 'Stake Ranked Episode 1', 'Stake Ranked Episode 2', 'StarLadder Budapest Major 2025', 'StarLadder StarSeries Fall 2025', 'StarLadder StarSeries Fall 2025. Closed Qualifier', 'Svenska Cupen 2023', 'Thunderpick World Championship 2023', 'Thunderpick World Championship 2024', 'Thunderpick World Championship 2024 - Qualifiers Europe #1', 'Thunderpick World Championship 2025. Closed Qualifier', 'YaLLa Compass 2024', 'YaLLa Compass Fall 2024']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['143', '152', '22', '2799', '4080'] | 2023-10-26 19:00:00 | 2023-10-27 17:30:00 | ['233392', '233623'] |
| 2 | ['143', '22', '2799', '315'] | 2023-10-28 11:00:00 | 2023-10-28 11:00:00 | ['233626'] |
| 3 | ['143', '152', '22', '2799', '4080'] | 2023-10-28 16:00:00 | 2023-11-11 10:00:00 | ['233394', '233525', '233810', '233814', '234100'] |
| 4 | ['143', '152', '22', '2799', '773'] | 2023-11-22 12:00:00 | 2023-11-23 12:00:00 | ['234114', '234121'] |
| 5 | ['143', '152', '22', '4080', '7292'] | 2023-12-02 20:55:00 | 2023-12-03 17:00:00 | ['235165', '235172'] |
| 6 | ['143', '152', '22', '2799', '773'] | 2024-01-13 13:00:00 | 2024-02-15 16:25:00 | ['236883', '236885', '237495', '237054', '237059', '237062', '239194', '239521', '239554'] |
| 7 | ['4080', '5114', '588', '7292', '7328'] | 2024-03-06 19:55:00 | 2024-03-06 19:55:00 | ['240218'] |
| 8 | ['143', '4080', '5114', '7292', '773'] | 2024-03-27 20:00:00 | 2024-03-27 22:20:00 | ['242512', '242628', '242634'] |
| 9 | ['143', '3396', '4080', '5114', '773'] | 2024-04-02 16:30:00 | 2024-04-18 19:00:00 | ['242670', '242679', '242801', '242682', '242691', '242684', '243073', '243078', '243083', '243740'] |
| 10 | ['3396', '4080', '5114', '773', '8252'] | 2024-04-30 17:30:00 | 2024-05-05 14:30:00 | ['243218', '243718', '243734', '243729', '243732'] |
| 11 | ['1203', '143', '3396', '4080', '773'] | 2024-05-16 21:15:00 | 2024-09-07 17:30:00 | ['244851', '244856', '244858', '245747', '245753', '245773', '245779', '245785', '248542', '248547', '248549', '249173', '249175', '249192', '251170', '250811', '251693', '251695', '251697', '251709', '253080', '253432', '253434', '253486', '252990', '253317', '254015', '249795', '254387', '254395', '254405'] |
| 12 | ['1203', '143', '3396', '4080', '588'] | 2024-10-04 12:00:00 | 2024-10-04 12:00:00 | ['256524'] |
| 13 | ['1203', '143', '3396', '588', '7328'] | 2024-10-16 15:40:00 | 2024-11-24 15:00:00 | ['256950', '256959', '257169', '257185', '257221', '258492', '258525', '257067', '259741', '259762', '259818', '259879', '259908'] |
| 14 | ['150', '3110', '3396', '3517', '709'] | 2025-02-05 13:00:00 | 2025-06-22 19:00:00 | ['263921', '263940', '263943', '266081', '266090', '266099', '266100', '267149', '267162', '267169', '267172', '267255', '267272', '267338', '267350', '267347', '267353', '267356', '269685', '270105', '270126', '270136', '270142', '270416', '271335', '271343', '271358', '271411', '271547', '271575', '272796', '272823', '272883', '272886'] |
| 15 | ['150', '3110', '3396', '709', '8252'] | 2025-07-23 19:30:00 | 2025-08-07 18:15:00 | ['273592', '274069', '273615', '274136', '274273', '274571'] |
| 16 | ['150', '3110', '3396', '3517', '709'] | 2025-08-11 17:00:00 | 2025-08-11 17:00:00 | ['275690'] |
| 17 | ['150', '3110', '3396', '709', '8252'] | 2025-08-13 16:00:00 | 2025-12-02 22:20:00 | ['275986', '275991', '276686', '276700', '277514', '277171', '277813', '277823', '277828', '277827', '280507', '280613', '280648', '281150', '281806', '282141', '282151', '282231', '282415', '282443', '282459', '282474'] |
| 18 | ['150', '20794', '3396', '709', '8252'] | 2026-01-15 19:00:00 | 2026-03-21 14:00:00 | ['283621', '284565', '285740', '285749', '286376', '286381', '286384', '287186', '287192', '287195', '287474', '287558', '287582', '287650', '287747', '288246', '288250', '288258', '288262', '288503', '288508', '288511', '288454', '288481'] |
| 19 | ['150', '20794', '3396', '709'] | 2026-04-01 20:30:00 | 2026-04-01 20:30:00 | ['289273'] |
| 20 | ['150', '20794', '3396', '709', '8252'] | 2026-04-02 20:45:00 | 2026-04-03 15:30:00 | ['289277', '289282'] |
| 21 | ['150', '20794', '484', '709', '8252'] | 2026-05-20 07:20:00 | 2026-05-30 19:30:00 | ['292633', '293081', '293785', '293802', '293805', '293798', '293807'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['143', '152', '22', '2799', '773'] (active 2024-01-13 13:00:00 to 2024-02-15 16:25:00) vs roster ['4080', '5114', '588', '7292', '7328'] (active 2024-03-06 19:55:00 to 2024-03-06 19:55:00) (20-day gap)

**Assessment**: (a) same organization with roster turnover
**Proposed decision**: `KEEP_AS_SINGLE_TEAM`
**Rationale**: 182 matches under this name with clean sequential roster eras (each era's appearances are chronologically disjoint from the next - a roster hand-off pattern, not simultaneous usage). Per the review's own rule, zero overlap between two eras of an established, high-volume name is NOT by itself evidence of a different organization - full roster turnover over a multi-year dataset is expected for real orgs. Recommend keeping as a single identity, with the era boundaries available for anyone who later wants roster-level (not org-level) granularity.

## `Rhyno Esports`

- First appearance: 2023-12-27 14:00:00  |  Last appearance: 2024-10-03 18:30:00
- Matches: 33  |  Tournaments (9): ['Betswap Winter Cup', 'CCT Season 2 European Series #11', 'CCT Season 2 European Series #13', 'CCT Season 2 European Series #4', 'ESL Challenger Atlanta 2024. Qualifier Europe', 'Elisa Invitational Fall 2024', 'IEM Rio 2024: Qualifier Europe', 'Perfect World Shanghai Major 2024: EU Qualifier', 'RES European Series #5']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['3109', '3122', '6617'] | 2023-12-27 14:00:00 | 2024-05-28 15:00:00 | ['235845', '235854', '245146', '245150', '245292', '245412', '245556'] |
| 2 | ['3109', '942'] | 2024-06-04 21:00:00 | 2024-08-20 15:00:00 | ['248336', '248341', '248343', '248955', '248959', '248961', '253084', '252871', '253343', '253423'] |
| 3 | ['3109', '3122', '3513', '6617'] | 2024-08-21 14:15:00 | 2024-08-23 12:00:00 | ['253057', '253623', '253678', '253736'] |
| 4 | ['3109', '942'] | 2024-08-28 15:00:00 | 2024-08-28 15:00:00 | ['253767'] |
| 5 | ['1365', '3122', '3513', '4322', '6617'] | 2024-09-02 17:10:00 | 2024-09-04 11:00:00 | ['254659', '254708'] |
| 6 | ['3109', '942'] | 2024-09-04 15:00:00 | 2024-09-04 15:00:00 | ['254381'] |
| 7 | ['1365', '3122', '3513', '4322', '6617'] | 2024-09-06 11:00:00 | 2024-09-08 18:30:00 | ['254889', '255115'] |
| 8 | ['2716', '3122', '3513', '6617'] | 2024-09-24 15:00:00 | 2024-10-03 18:30:00 | ['255689', '255827', '256151', '256207', '256319', '256517'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['3109', '942'] (active 2024-06-04 21:00:00 to 2024-08-20 15:00:00) vs roster ['1365', '3122', '3513', '4322', '6617'] (active 2024-09-02 17:10:00 to 2024-09-04 11:00:00) (13-day gap)
- roster ['3109', '942'] (active 2024-06-04 21:00:00 to 2024-08-20 15:00:00) vs roster ['1365', '3122', '3513', '4322', '6617'] (active 2024-09-06 11:00:00 to 2024-09-08 18:30:00) (16-day gap)
- roster ['3109', '942'] (active 2024-06-04 21:00:00 to 2024-08-20 15:00:00) vs roster ['2716', '3122', '3513', '6617'] (active 2024-09-24 15:00:00 to 2024-10-03 18:30:00) (35-day gap)
- roster ['3109', '942'] (active 2024-08-28 15:00:00 to 2024-08-28 15:00:00) vs roster ['1365', '3122', '3513', '4322', '6617'] (active 2024-09-02 17:10:00 to 2024-09-04 11:00:00) (5-day gap)
- roster ['3109', '942'] (active 2024-08-28 15:00:00 to 2024-08-28 15:00:00) vs roster ['1365', '3122', '3513', '4322', '6617'] (active 2024-09-06 11:00:00 to 2024-09-08 18:30:00) (8-day gap)
- roster ['3109', '942'] (active 2024-08-28 15:00:00 to 2024-08-28 15:00:00) vs roster ['2716', '3122', '3513', '6617'] (active 2024-09-24 15:00:00 to 2024-10-03 18:30:00) (27-day gap)
- roster ['1365', '3122', '3513', '4322', '6617'] (active 2024-09-02 17:10:00 to 2024-09-04 11:00:00) vs roster ['3109', '942'] (active 2024-09-04 15:00:00 to 2024-09-04 15:00:00) (0-day gap)
- roster ['3109', '942'] (active 2024-09-04 15:00:00 to 2024-09-04 15:00:00) vs roster ['1365', '3122', '3513', '4322', '6617'] (active 2024-09-06 11:00:00 to 2024-09-08 18:30:00) (1-day gap)
- roster ['3109', '942'] (active 2024-09-04 15:00:00 to 2024-09-04 15:00:00) vs roster ['2716', '3122', '3513', '6617'] (active 2024-09-24 15:00:00 to 2024-10-03 18:30:00) (20-day gap)

**Assessment**: (c) insufficient evidence
**Proposed decision**: `MANUAL_REVIEW`
**Rationale**: Only 33 matches under this name; roster eras are sequential (consistent with turnover) but the sample is too small to be confident this isn't actually two different low-tier squads that happened to reuse a common-sounding name. Needs manual review.

## `SPARTA Esports`

- First appearance: 2025-08-01 18:00:00  |  Last appearance: 2026-06-17 18:00:00
- Matches: 66  |  Tournaments (20): ['CCT Europe 2026 Series #1', 'CCT Europe 2026 Series #4', 'CCT Season 3 European Series #10', 'CCT Season 3 European Series #11', 'CCT Season 3 European Series #12', 'CCT Season 3 European Series #5', 'CCT Season 3 European Series #7', 'CCT Season 3 European Series #9', 'CIS LAN #5', 'CIS LAN Championship #3', 'CIS LAN Championship #4', 'ESL Challenger League Season 50: Europe — Cup #3', 'ESL Challenger League Season 51: Europe — Cup #4', 'European Pro League Series 3', 'European Pro League Series 7', 'Galaxy Battle 2025 // Phase 5', 'Majestic LanDaLan #3. Closed Qualifier', 'NODWIN Clutch Series #7', 'RES Showdown Fall 2025', 'Thunderpick World Championship 2025: European Series #2']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['3323', '4549', '4792', '8185', '8188'] | 2025-08-01 18:00:00 | 2025-08-01 18:00:00 | ['274537'] |
| 2 | ['3323', '4792', '8185', '8188'] | 2025-08-09 11:00:00 | 2025-08-27 21:15:00 | ['275325', '275330', '275333', '275647', '276437', '276481', '276447', '276563', '276587', '276813', '276809', '276855'] |
| 3 | ['3323', '4549', '4792', '8185', '8188'] | 2025-09-19 17:30:00 | 2025-09-23 11:30:00 | ['278009', '278235', '278446'] |
| 4 | ['3323', '4792', '8185', '8188'] | 2025-10-06 20:00:00 | 2025-12-12 12:00:00 | ['279383', '279412', '279522', '280531', '280543', '280577', '280623', '280662', '280733', '280809', '280974', '281094', '281569', '281596', '281607', '282203', '282208', '282577', '282537', '282745', '282799', '282843', '282895', '283185', '283227'] |
| 5 | ['1892', '4792', '8185', '8188', '961'] | 2026-03-10 18:30:00 | 2026-03-10 21:15:00 | ['288137', '288142'] |
| 6 | ['1008', '1436', '3464', '932', '936'] | 2026-04-02 16:45:00 | 2026-05-25 16:40:00 | ['290034', '290038', '290042', '292016', '291526', '292053', '291525', '292139', '292142', '292137', '292292', '292291', '292932', '293056', '293350', '293554', '293559'] |
| 7 | ['1436', '4792', '8188', '932', '936'] | 2026-06-10 11:00:00 | 2026-06-17 18:00:00 | ['294404', '294690', '294910', '295106', '294968', '295110'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['1892', '4792', '8185', '8188', '961'] (active 2026-03-10 18:30:00 to 2026-03-10 21:15:00) vs roster ['1008', '1436', '3464', '932', '936'] (active 2026-04-02 16:45:00 to 2026-05-25 16:40:00) (22-day gap)

**Assessment**: (a) same organization with roster turnover
**Proposed decision**: `KEEP_AS_SINGLE_TEAM`
**Rationale**: 66 matches under this name with clean sequential roster eras (each era's appearances are chronologically disjoint from the next - a roster hand-off pattern, not simultaneous usage). Per the review's own rule, zero overlap between two eras of an established, high-volume name is NOT by itself evidence of a different organization - full roster turnover over a multi-year dataset is expected for real orgs. Recommend keeping as a single identity, with the era boundaries available for anyone who later wants roster-level (not org-level) granularity.

## `Sashi Esport`

- First appearance: 2024-04-01 13:00:00  |  Last appearance: 2026-06-25 23:00:00
- Matches: 198  |  Tournaments (49): ['CCT Season 2 European Series #16', 'CCT Season 2 European Series #17', 'CCT Season 2 European Series #18', 'CCT Season 2 European Series #19', 'CCT Season 2 European Series #20', 'CCT Season 3 European Series #12', 'CCT Season 3 European Series #2', 'CCT Season 3 European Series #4', 'CCT Season 3 European Series #6', 'CCT Season 3 European Series #7', 'CCT Season 3 European Series #8', 'CCT Season 3 European Series #9', 'CS2 Asia Championships 2025. Qualifier', 'DraculaN Season 6', 'ESL Challenger Atlanta 2024. Qualifier Europe', 'ESL Challenger League Season 48 — Europe', 'ESL Challenger League Season 49 — Europe', 'ESL Challenger League Season 50: Europe — Cup #1', 'ESL Challenger League Season 50: Europe — Cup #2', 'ESL Challenger League Season 50: Europe — Cup #3', 'ESL Pro League Season 22. European Qualifier', 'Elisa Invitational Fall 2024', 'Elisa Invitational Spring 2024', 'Esports World Cup 2024 по CS2', 'Esports World Cup 2024 по CS2. Closed Qualifier', 'Esports World Cup 2024 по CS2. Open Qualifier', 'European Pro League Series 2', 'European Pro League Series 4', 'European Pro League Series 7', 'Exort The Proving Grounds Season 5', 'ICE Invitational 2025', 'IEM Rio 2024: Qualifier Europe', 'NODWIN Clutch Series #4', 'Nordic Masters Fall 2024', 'PGL Astana 2025. Closed European Qualifier', 'PGL Bucharest 2025. European Qualifiers', 'Parken Challenger Championship #1', 'Perfect World Shanghai Major 2024: EU Qualifier', 'Perfect World Shanghai Major 2024: European RMR B', 'RES European Series #4', 'Red Bull GIBAWAY JOURNEY 2025', 'Skyesports Championship 2024: European Qualifier', 'Super DraculaN Season 1', 'Thunderpick World Championship 2025: European Series #1', 'YaLLa Compass 2024', 'YaLLa Compass Fall 2024', 'YaLLa Compass Spring 2024', 'YaLLa Compass Summer 2024', 'YaLLa Compass Winter 2025']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['1446', '1689', '2053', '2537', '3922'] | 2024-04-01 13:00:00 | 2024-04-09 13:00:00 | ['242730', '243107'] |
| 2 | ['23', '24', '2847', '3933', '4451'] | 2024-04-11 12:00:00 | 2024-04-11 12:00:00 | ['243346'] |
| 3 | ['1446', '1689', '2053', '2537', '3922'] | 2024-04-15 15:00:00 | 2024-11-24 11:00:00 | ['243482', '243503', '243752', '243748', '243882', '243886', '243908', '243911', '243917', '243914', '244003', '244008', '244113', '244234', '244242', '244246', '244248', '244657', '244839', '244666', '244843', '244672', '244675', '244847', '245002', '245745', '245750', '245755', '245777', '245786', '249248', '249253', '250259', '250825', '250830', '252220', '252224', '252228', '253456', '253458', '253482', '253076', '253651', '253705', '253747', '253775', '254074', '253315', '254018', '254020', '254540', '255092', '254918', '254923', '254925', '255232', '255378', '256895', '256910', '256917', '258388', '258410', '258428', '258444', '258447', '257079', '259750', '259764', '259811', '259884'] |
| 4 | ['1689', '2053', '2537', '2848', '3785'] | 2025-01-13 12:00:00 | 2025-02-28 17:00:00 | ['262238', '261822', '261829', '261831', '262934', '263250', '263069', '263438', '263022', '263458', '263472', '263575', '263807', '264019', '264046', '264318', '264808', '264356', '264880', '264777', '265012', '265073', '265126', '263081', '265327', '263078', '265506', '263089', '265653', '265676'] |
| 5 | ['153', '1689', '2053', '2537', '2848'] | 2025-03-02 13:00:00 | 2025-03-02 13:00:00 | ['265744'] |
| 6 | ['1689', '2053', '2537', '2848', '3785'] | 2025-03-03 12:00:00 | 2025-03-18 22:00:00 | ['265835', '265828', '266162', '266142', '266334', '266111', '266119', '266123', '266449', '266479', '266481', '265758'] |
| 7 | ['153', '1689', '2053', '2537', '2848'] | 2025-03-27 15:00:00 | 2025-03-29 15:00:00 | ['267232', '267283', '267295'] |
| 8 | ['1689', '2053', '2537', '2848', '3785'] | 2025-04-01 20:45:00 | 2025-05-05 11:00:00 | ['266801', '269835'] |
| 9 | ['1689', '2053', '2537', '2848', '8256'] | 2025-05-20 11:00:00 | 2025-05-26 14:30:00 | ['271321', '271337', '271359', '271419'] |
| 10 | ['1689', '2053', '2848', '3457', '8256'] | 2025-07-14 12:00:00 | 2025-07-16 21:00:00 | ['273769', '273770', '273793', '273812', '273822', '273825'] |
| 11 | ['1689', '2053', '2848', '3922', '8256'] | 2025-07-29 20:05:00 | 2025-08-08 20:00:00 | ['274391', '274452', '274585', '274955', '275273'] |
| 12 | ['1689', '2053', '2537', '2848', '3785'] | 2025-08-10 18:30:00 | 2025-08-10 18:30:00 | ['275679'] |
| 13 | ['1689', '2053', '2848', '3922', '8256'] | 2025-08-12 20:00:00 | 2025-10-10 20:00:00 | ['275718', '275815', '276049', '276145', '277159', '277417', '277627', '277704', '277952', '277912', '277993', '277968', '278035', '278038', '278888', '278898', '279370', '279406', '279523', '279592', '279688'] |
| 14 | ['2053', '2848', '3922', '8256'] | 2025-10-11 14:00:00 | 2025-10-11 14:00:00 | ['279846'] |
| 15 | ['1689', '2053', '2848', '3922', '8256'] | 2025-10-11 20:00:00 | 2025-10-21 11:00:00 | ['279697', '280518'] |
| 16 | ['166', '2053', '2848', '3922', '8256'] | 2025-10-24 11:00:00 | 2025-10-24 11:00:00 | ['280542'] |
| 17 | ['1689', '2053', '2848', '3922', '8256'] | 2025-10-26 18:00:00 | 2025-10-30 12:00:00 | ['280566', '280628', '280723'] |
| 18 | ['166', '2053', '2848', '3922', '8256'] | 2025-11-07 14:00:00 | 2025-12-10 18:00:00 | ['281238', '281246', '281288', '281312', '281337', '281366', '281375', '281381', '282646', '282744', '282850', '282890', '283182'] |
| 19 | ['2053', '2848', '3922', '641', '8256'] | 2026-01-30 12:00:00 | 2026-02-23 15:00:00 | ['285808', '285814', '286058', '287307', '287314', '287317', '287325', '287328', '287335', '287340'] |
| 20 | ['1266', '2053', '3922', '641', '8256'] | 2026-03-31 14:00:00 | 2026-04-01 16:00:00 | ['289689', '289696', '289707', '289711'] |
| 21 | ['2053', '2848', '3922', '641', '8256'] | 2026-06-16 11:00:00 | 2026-06-25 23:00:00 | ['294962', '294974', '295795', '295793', '295832', '295835'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['1446', '1689', '2053', '2537', '3922'] (active 2024-04-01 13:00:00 to 2024-04-09 13:00:00) vs roster ['23', '24', '2847', '3933', '4451'] (active 2024-04-11 12:00:00 to 2024-04-11 12:00:00) (1-day gap)
- roster ['23', '24', '2847', '3933', '4451'] (active 2024-04-11 12:00:00 to 2024-04-11 12:00:00) vs roster ['1446', '1689', '2053', '2537', '3922'] (active 2024-04-15 15:00:00 to 2024-11-24 11:00:00) (4-day gap)

**Assessment**: (a) same organization with roster turnover
**Proposed decision**: `KEEP_AS_SINGLE_TEAM`
**Rationale**: 198 matches under this name with clean sequential roster eras (each era's appearances are chronologically disjoint from the next - a roster hand-off pattern, not simultaneous usage). Per the review's own rule, zero overlap between two eras of an established, high-volume name is NOT by itself evidence of a different organization - full roster turnover over a multi-year dataset is expected for real orgs. Recommend keeping as a single identity, with the era boundaries available for anyone who later wants roster-level (not org-level) granularity.

## `Team 2S`

- First appearance: 2024-12-27 18:35:00  |  Last appearance: 2024-12-27 19:35:00
- Matches: 2  |  Tournaments (1): ['BetBoom Aunkere Cup 2x2']
- Generic-name pattern match: True

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

_No appearances with a known roster - cannot build an era timeline._

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- None found at the era level in this deeper, name-specific re-check.

**Assessment**: (b) likely recycled/ambiguous team name
**Proposed decision**: `EXCLUDE_FROM_IDENTITY_DEPENDENT_FEATURES`
**Rationale**: Generic/placeholder-looking name with a low match count - consistent with an ad-hoc qualifier stand-in roster rather than a persistent organization.

## `Team 3DMAX`

- First appearance: 2023-11-24 14:50:00  |  Last appearance: 2026-05-29 13:00:00
- Matches: 255  |  Tournaments (64): ['BLAST Bounty Spring 2025. Closed Qualifiers', 'BLAST Bounty Winter 2026. Closed Qualifiers', 'BLAST.tv Austin Major 2025', 'BetBoom Dacha Belgrade 2024 #2. European Qualifier', 'CCT Season 2 European Series #12', 'CCT Season 2 European Series #13', 'CCT Season 2 European Series #14', 'CCT Season 2 European Series #3', 'CCT Season 2 European Series #4', 'CCT Season 2 European Series #5', 'CCT Season 2 European Series #8', 'CS Asia Championships 2026', 'CS2 Asia Championships 2025', 'ESL Challenger Jonköping 2023', 'ESL Challenger Jonköping 2024. Open Qualifier', 'ESL Challenger Katowice 2024. Closed Qualifier', 'ESL Challenger League Season 46 — Europe', 'ESL Challenger League Season 47 — Europe', 'ESL Challenger League Season 48 — Europe', 'ESL Pro League Season 19', 'ESL Pro League Season 20', 'ESL Pro League Season 20: European Conference', 'ESL Pro League Season 21', 'ESL Pro League Season 22', 'ESL Pro League Season 23', 'Elisa Invitational Fall 2024', 'Esports World Cup 2025 по CS2', 'FISSURE PLAYGROUND 1 — CS', 'FISSURE PLAYGROUND 2 — CS', 'IEM Chengdu 2024. Open Qualifier', 'IEM Chengdu 2025', 'IEM Cologne 2024', 'IEM Cologne 2025', 'IEM Dallas 2024: Qualifier Europe', 'IEM Dallas 2025', 'IEM Dallas 2025: Qualifier Europe', 'IEM Katowice 2025', 'IEM Krakow 2026', 'IEM Melbourne 2025', 'IEM Rio 2024: Qualifier Europe', 'IEM Rio 2026', 'PGL Bucharest 2025', 'PGL Bucharest 2026', 'PGL CS2 Major Copenhagen 2024: Closed Qualifiers', 'PGL CS2 Major Copenhagen 2024: European RMR A', 'PGL Cluj-Napoca 2025', 'PGL Cluj-Napoca 2026', 'PGL Masters Bucharest 2025', 'Perfect World CS Challenge Series #1', 'Perfect World Shanghai Major 2024', 'Perfect World Shanghai Major 2024: European RMR B', 'RES European Series #1', 'RES European Series #4', 'RES Showdown Spring 2026', 'RES Western European Masters: Spring 2024', 'Regional Clash Arena Europe', 'Skyesports Championship 2024', 'Skyesports Championship 2024: European Qualifier', 'Skyesports Masters 2024: Qualifier', 'Stake Ranked Episode 2', 'StarLadder Budapest Major 2025', 'Thunderpick World Championship 2024', 'YaLLa Compass Spring 2024', 'YaLLa Compass Summer 2024']
- Generic-name pattern match: True

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['2894', '3072', '413', '499', '656'] | 2023-11-24 14:50:00 | 2024-03-13 18:00:00 | ['235024', '235029', '235055', '235070', '235071', '237447', '237704', '237725', '237879', '237932', '237979', '237985', '238831', '239203', '239208', '239210', '239223', '239199', '239527', '239558', '239566', '240273', '240278', '240481', '240595', '240058'] |
| 2 | ['1486', '2894', '3072', '499', '656'] | 2024-03-15 18:30:00 | 2024-03-15 18:30:00 | ['240035'] |
| 3 | ['20', '2894', '3072', '499', '656'] | 2024-03-19 16:20:00 | 2024-03-21 17:00:00 | ['241885', '241893'] |
| 4 | ['2894', '3072', '4011', '499', '656'] | 2024-03-27 20:00:00 | 2024-03-27 20:00:00 | ['242501'] |
| 5 | ['1486', '2894', '3072', '499', '656'] | 2024-03-28 17:00:00 | 2024-03-28 17:00:00 | ['240044'] |
| 6 | ['2894', '3072', '4011', '499', '656'] | 2024-04-02 16:00:00 | 2024-04-02 16:00:00 | ['242737'] |
| 7 | ['1486', '2894', '3072', '499', '656'] | 2024-04-03 21:00:00 | 2024-04-10 21:00:00 | ['240071', '240066', '243149', '243374'] |
| 8 | ['2894', '3072', '4011', '499', '656'] | 2024-04-11 16:00:00 | 2025-02-03 21:40:00 | ['243101', '243505', '243995', '243197', '243642', '243643', '244231', '244239', '244245', '244417', '244829', '244833', '245123', '245273', '248238', '248240', '245376', '245380', '245393', '248966', '248974', '248982', '248987', '248997', '249007', '249017', '249021', '249023', '249246', '249099', '249226', '249252', '249241', '249364', '249105', '249385', '249238', '249108', '251286', '251301', '251308', '251321', '250770', '250775', '250778', '251594', '251599', '251609', '251613', '251448', '251971', '253186', '253855', '253865', '253872', '249803', '254415', '254418', '254424', '254429', '254430', '254912', '255584', '255796', '255595', '255598', '255818', '256164', '256170', '256157', '256173', '256184', '256516', '256532', '256602', '256718', '256804', '256896', '256901', '256916', '256919', '256921', '257159', '257176', '257578', '257607', '257616', '257080', '259748', '259770', '260138', '260278', '260285', '260321', '261932', '262738', '263538', '262757', '263588', '263599'] |
| 9 | ['18', '3072', '4011', '499', '656'] | 2025-02-10 16:30:00 | 2025-09-16 12:00:00 | ['264380', '264496', '264540', '265025', '265060', '265112', '265256', '265568', '265698', '265807', '265918', '266185', '266387', '266425', '266459', '266511', '267875', '268220', '268255', '268433', '268558', '269015', '269031', '270060', '271013', '271017', '271192', '271206', '271230', '271239', '271261', '271274', '271294', '271308', '273544', '273548', '273552', '273837', '273590', '274068', '274116', '273609', '274129', '274265', '275403', '276488', '276682', '276912', '276922', '277225', '277348', '277374', '277386'] |
| 10 | ['18', '26221', '3072', '4011', '656'] | 2025-09-28 14:00:00 | 2025-09-28 14:00:00 | ['278564'] |
| 11 | ['18', '3072', '4011', '499', '656'] | 2025-09-29 11:30:00 | 2026-02-16 16:15:00 | ['278914', '279009', '279188', '279269', '279287', '279438', '279494', '279519', '279562', '279871', '279902', '279914', '279917', '279920', '279925', '279928', '280510', '280602', '280644', '280678', '280565', '280959', '282228', '282418', '282435', '282463', '282475', '282657', '282666', '282766', '282780', '283628', '284571', '285737', '284610', '286017', '286026', '286372', '286355', '286966', '286994'] |
| 12 | ['1486', '3072', '4011', '499', '656'] | 2026-03-01 22:00:00 | 2026-05-29 13:00:00 | ['287472', '287554', '287578', '287644', '287748', '287801', '288057', '288087', '288163', '288205', '289906', '290060', '290105', '290129', '290331', '290338', '290344', '289932', '290407', '290411', '292626', '293082', '293780', '293788'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- None found at the era level in this deeper, name-specific re-check.

**Assessment**: (a) same organization with roster turnover
**Proposed decision**: `KEEP_AS_SINGLE_TEAM`
**Rationale**: Matched the generic-name regex on spelling alone (e.g. 'Team <digits>' pattern), but this deeper pass finds no close-in-time zero-roster-overlap evidence at all and a high match count - likely a regex false positive on a real, established team name, not a placeholder identity.

## `Team 7AM`

- First appearance: 2024-01-30 20:00:00  |  Last appearance: 2024-01-30 20:00:00
- Matches: 1  |  Tournaments (1): ['IEM Chengdu 2024. Open Qualifier']
- Generic-name pattern match: True

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['1333', '1442', '2798', '4120'] | 2024-01-30 20:00:00 | 2024-01-30 20:00:00 | ['238837'] |

Eras chronologically sequential (no overlap between different rosters' active windows): n/a (only one roster era)

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- None found at the era level in this deeper, name-specific re-check.

**Assessment**: (b) likely recycled/ambiguous team name
**Proposed decision**: `EXCLUDE_FROM_IDENTITY_DEPENDENT_FEATURES`
**Rationale**: Generic/placeholder-looking name with a low match count - consistent with an ad-hoc qualifier stand-in roster rather than a persistent organization.

## `Team Buster`

- First appearance: 2024-12-20 16:00:00  |  Last appearance: 2026-05-16 11:00:00
- Matches: 26  |  Tournaments (5): ['BetBoom Streamers Battle CS2', 'BetBoom Streamers Battle x Динамо CS #4', 'PGL Astana 2026', 'Rofl Streamers Battle', 'Streamers League #1']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['3395', '4543', '6900'] | 2024-12-20 16:00:00 | 2024-12-22 22:00:00 | ['261052', '261059', '261064', '261081', '261083'] |
| 2 | ['3395', '6900', '8055'] | 2024-12-23 23:00:00 | 2024-12-24 18:00:00 | ['261088', '261091', '261093'] |
| 3 | ['4543', '7582'] | 2025-12-18 16:30:00 | 2025-12-28 12:30:00 | ['283342', '283350', '283354', '283356', '283489', '283497', '283499'] |
| 4 | ['1002', '130'] | 2026-03-16 14:00:00 | 2026-03-17 21:30:00 | ['288542', '288548', '288550', '288553'] |
| 5 | ['7582'] | 2026-04-21 17:50:00 | 2026-04-25 18:00:00 | ['291230', '291233', '291235', '291368', '291370', '291379'] |
| 6 | ['392', '608', '7582'] | 2026-05-16 11:00:00 | 2026-05-16 11:00:00 | ['293238'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['1002', '130'] (active 2026-03-16 14:00:00 to 2026-03-17 21:30:00) vs roster ['7582'] (active 2026-04-21 17:50:00 to 2026-04-25 18:00:00) (34-day gap)
- roster ['1002', '130'] (active 2026-03-16 14:00:00 to 2026-03-17 21:30:00) vs roster ['392', '608', '7582'] (active 2026-05-16 11:00:00 to 2026-05-16 11:00:00) (59-day gap)

**Assessment**: (c) insufficient evidence
**Proposed decision**: `MANUAL_REVIEW`
**Rationale**: Only 26 matches under this name; roster eras are sequential (consistent with turnover) but the sample is too small to be confident this isn't actually two different low-tier squads that happened to reuse a common-sounding name. Needs manual review.

## `Team CS2NEWS`

- First appearance: 2024-12-20 15:00:00  |  Last appearance: 2025-12-28 18:15:00
- Matches: 25  |  Tournaments (5): ['BetBoom Streamers Battle CS #3', 'BetBoom Streamers Battle CS2', 'BetBoom Streamers Battle CS2 #2', 'BetBoom Streamers Battle x Динамо CS #4', 'BetBoom ct0m Cup']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['4627', '581', '7582'] | 2024-12-20 15:00:00 | 2024-12-20 15:00:00 | ['261049'] |
| 2 | ['2988', '581', '7582'] | 2024-12-20 19:05:00 | 2024-12-20 19:05:00 | ['261056'] |
| 3 | ['1342', '7582'] | 2025-04-26 15:15:00 | 2025-05-02 17:30:00 | ['269248', '269251', '269260', '269267', '269275', '269290', '269293', '269588', '269593'] |
| 4 | ['579'] | 2025-08-05 20:20:00 | 2025-08-05 22:00:00 | ['275212', '275217'] |
| 5 | ['1092', '27103', '392'] | 2025-08-10 21:35:00 | 2025-08-17 13:00:00 | ['275518', '275523', '275531', '276154'] |
| 6 | ['27035'] | 2025-12-18 22:50:00 | 2025-12-28 18:15:00 | ['283346', '283352', '283357', '283361', '283483', '283494', '283498', '283501'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['579'] (active 2025-08-05 20:20:00 to 2025-08-05 22:00:00) vs roster ['1092', '27103', '392'] (active 2025-08-10 21:35:00 to 2025-08-17 13:00:00) (4-day gap)

**Assessment**: (c) insufficient evidence
**Proposed decision**: `MANUAL_REVIEW`
**Rationale**: Only 25 matches under this name; roster eras are sequential (consistent with turnover) but the sample is too small to be confident this isn't actually two different low-tier squads that happened to reuse a common-sounding name. Needs manual review.

## `Team Shadowkek`

- First appearance: 2023-12-21 14:45:00  |  Last appearance: 2025-12-27 18:05:00
- Matches: 36  |  Tournaments (6): ['BetBoom All-Star 5x5', 'BetBoom Streamers Battle CS #3', 'BetBoom Streamers Battle CS2', 'BetBoom Streamers Battle CS2 #2', 'BetBoom Streamers Battle x Динамо CS #4', 'BetBoom ct0m Cup']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['3059'] | 2023-12-21 14:45:00 | 2023-12-21 20:00:00 | ['235743', '235746'] |
| 2 | ['1954', '6171'] | 2024-12-20 15:00:00 | 2024-12-23 19:00:00 | ['261048', '261054', '261062', '261067', '261080', '261087'] |
| 3 | ['3059', '6693'] | 2025-04-26 17:45:00 | 2025-05-04 18:00:00 | ['269253', '269255', '269263', '269264', '269274', '269295', '269298', '269582', '269592', '269688', '269691', '269693'] |
| 4 | ['6171'] | 2025-08-05 14:40:00 | 2025-08-05 16:00:00 | ['275202', '275207'] |
| 5 | ['3059', '6693'] | 2025-08-13 15:00:00 | 2025-08-19 20:00:00 | ['275526', '275528', '275534', '276148', '276220', '276264', '276263'] |
| 6 | ['26759', '6693'] | 2025-12-19 21:30:00 | 2025-12-27 18:05:00 | ['283326', '283328', '283330', '283340', '283482', '283486', '283496'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['6171'] (active 2025-08-05 14:40:00 to 2025-08-05 16:00:00) vs roster ['3059', '6693'] (active 2025-08-13 15:00:00 to 2025-08-19 20:00:00) (7-day gap)

**Assessment**: (c) insufficient evidence
**Proposed decision**: `MANUAL_REVIEW`
**Rationale**: Only 36 matches under this name; roster eras are sequential (consistent with turnover) but the sample is too small to be confident this isn't actually two different low-tier squads that happened to reuse a common-sounding name. Needs manual review.

## `Team Spirit`

- First appearance: 2023-10-06 20:00:00  |  Last appearance: 2026-06-20 20:00:00
- Matches: 210  |  Tournaments (50): ['BLAST Bounty Fall 2025', 'BLAST Bounty Fall 2025. Closed Qualifiers', 'BLAST Bounty Spring 2025', 'BLAST Bounty Spring 2025. Closed Qualifiers', 'BLAST Bounty Winter 2026', 'BLAST Bounty Winter 2026. Closed Qualifiers', 'BLAST Open Lisbon 2025', 'BLAST Open London 2025. Closed Qualifiers', 'BLAST Open Spring 2026', 'BLAST Premier: Fall Final 2024', 'BLAST Premier: Fall Groups 2024', 'BLAST Premier: Spring Final 2024', 'BLAST Premier: Spring Groups 2024', 'BLAST Premier: Spring Showdown 2024', 'BLAST Premier: World Final 2024', 'BLAST Rivals Fall 2025', 'BLAST Rivals Spring 2025', 'BLAST.tv Austin Major 2025', 'BetBoom Dacha', 'BetBoom Dacha Belgrade 2024 #2', 'BetBoom Dacha CS2 Belgrade 2024', 'BetBoom Dacha. Closed Qualifier', 'CCT North Europe Series #8', 'ESL Challenger Jonköping 2023. Qualifiers', 'ESL Challenger League Season 46 — Europe', 'ESL Pro League Season 20', 'ESL Pro League Season 21', 'ESL Pro League Season 22', 'ESL Pro League Season 23', 'Esports World Cup 2024 по CS2', 'Esports World Cup 2025 по CS2', 'IEM Chengdu 2025', 'IEM Cologne 2024', 'IEM Cologne 2025', 'IEM Cologne Major 2026', 'IEM Dallas 2024', 'IEM Katowice 2024', 'IEM Katowice 2025', 'IEM Krakow 2026', 'IEM Rio 2026', 'PGL Astana 2025', 'PGL Astana 2026', 'PGL CS2 Major Copenhagen 2024: Closed Qualifiers', 'PGL CS2 Major Copenhagen 2024: European RMR B', 'PGL Major Copenhagen 2024', 'Perfect World Shanghai Major 2024', 'Perfect World Shanghai Major 2024: European RMR B', 'Roobet Cup 2023', 'StarLadder Budapest Major 2025', 'Thunderpick World Championship 2023']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['1003', '295', '3397', '4710', '5057'] | 2023-10-06 20:00:00 | 2023-12-06 17:00:00 | ['232899', '232915', '232921', '232934', '232937', '232969', '232963', '233279', '233311', '233403', '233611', '233618', '233380', '233516', '233641', '233647', '233285', '233859', '233927', '234027', '234164', '234165', '234335', '235250'] |
| 2 | ['5420', '5504', '5962', '6171', '7440'] | 2023-12-06 21:00:00 | 2023-12-06 21:00:00 | ['235053'] |
| 3 | ['1003', '295', '3397', '4710', '5057'] | 2023-12-07 12:00:00 | 2023-12-07 12:00:00 | ['235254'] |
| 4 | ['5420', '5504', '5962', '6171', '7440'] | 2023-12-07 21:00:00 | 2023-12-07 21:00:00 | ['235067'] |
| 5 | ['1003', '295', '3397', '4710', '5057'] | 2023-12-08 16:10:00 | 2023-12-10 13:55:00 | ['235255', '235480', '235483'] |
| 6 | ['1003', '295', '3397', '4710', '965'] | 2024-01-18 18:00:00 | 2024-01-20 18:00:00 | ['237455', '237691', '237734', '237898'] |
| 7 | ['295', '3155', '3397', '6171', '965'] | 2024-01-24 16:35:00 | 2024-01-25 16:10:00 | ['237043', '237048'] |
| 8 | ['1003', '295', '3397', '4710', '965'] | 2024-01-31 19:55:00 | 2025-06-19 22:00:00 | ['237909', '238872', '237920', '238991', '238998', '239149', '239183', '239309', '239731', '239749', '239771', '240225', '240235', '240240', '241907', '241935', '242002', '242237', '244365', '244375', '244877', '244880', '244893', '245539', '245542', '245835', '248162', '248171', '249360', '249363', '250260', '250838', '250800', '251674', '251680', '251464', '252001', '253932', '253937', '253939', '253952', '249798', '254417', '254421', '254425', '255355', '255339', '255347', '255124', '255815', '255875', '257147', '257151', '257619', '257623', '257626', '257068', '259739', '259757', '259810', '259883', '259907', '260130', '260268', '260281', '260314', '260386', '260393', '260396', '261916', '262772', '262942', '263271', '263273', '262758', '263604', '263794', '263797', '263871', '263883', '263799', '266170', '266390', '266426', '266529', '266533', '265546', '265774', '265786', '265789', '265764', '265768', '269210', '269230', '269214', '269219', '269678', '270104', '270117', '270417', '270675', '270677', '271258', '271276', '271288', '272616'] |
| 9 | ['16113', '295', '3397', '4710', '965'] | 2025-07-27 14:30:00 | 2025-09-01 22:10:00 | ['273616', '274135', '274269', '274282', '274285', '274550', '275386', '276044', '276137', '276138', '275412', '276519', '276539', '276546', '276550'] |
| 10 | ['16113', '295', '3395', '4710', '965'] | 2025-10-04 11:30:00 | 2025-12-13 19:00:00 | ['279256', '279276', '279428', '279556', '280557', '280941', '280950', '280954', '280632', '281465', '281475', '282652', '282669', '282761', '282872', '282880'] |
| 11 | ['1003', '3395', '3397', '4710', '965'] | 2026-01-16 15:55:00 | 2026-06-20 20:00:00 | ['283618', '284269', '284742', '284607', '286007', '286009', '286290', '286296', '287800', '288054', '288080', '288317', '288452', '288474', '288482', '288486', '289930', '290400', '290408', '290412', '290435', '290431', '290438', '292466', '292705', '292827', '293114', '293121', '293124', '294124', '294152', '294164', '294489', '294808', '294872', '294992', '295000'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['1003', '295', '3397', '4710', '5057'] (active 2023-10-06 20:00:00 to 2023-12-06 17:00:00) vs roster ['5420', '5504', '5962', '6171', '7440'] (active 2023-12-06 21:00:00 to 2023-12-06 21:00:00) (0-day gap)
- roster ['1003', '295', '3397', '4710', '5057'] (active 2023-10-06 20:00:00 to 2023-12-06 17:00:00) vs roster ['5420', '5504', '5962', '6171', '7440'] (active 2023-12-07 21:00:00 to 2023-12-07 21:00:00) (1-day gap)
- roster ['5420', '5504', '5962', '6171', '7440'] (active 2023-12-06 21:00:00 to 2023-12-06 21:00:00) vs roster ['1003', '295', '3397', '4710', '5057'] (active 2023-12-07 12:00:00 to 2023-12-07 12:00:00) (0-day gap)
- roster ['5420', '5504', '5962', '6171', '7440'] (active 2023-12-06 21:00:00 to 2023-12-06 21:00:00) vs roster ['1003', '295', '3397', '4710', '5057'] (active 2023-12-08 16:10:00 to 2023-12-10 13:55:00) (1-day gap)
- roster ['5420', '5504', '5962', '6171', '7440'] (active 2023-12-06 21:00:00 to 2023-12-06 21:00:00) vs roster ['1003', '295', '3397', '4710', '965'] (active 2024-01-18 18:00:00 to 2024-01-20 18:00:00) (42-day gap)
- roster ['5420', '5504', '5962', '6171', '7440'] (active 2023-12-06 21:00:00 to 2023-12-06 21:00:00) vs roster ['1003', '295', '3397', '4710', '965'] (active 2024-01-31 19:55:00 to 2025-06-19 22:00:00) (55-day gap)
- roster ['1003', '295', '3397', '4710', '5057'] (active 2023-12-07 12:00:00 to 2023-12-07 12:00:00) vs roster ['5420', '5504', '5962', '6171', '7440'] (active 2023-12-07 21:00:00 to 2023-12-07 21:00:00) (0-day gap)
- roster ['5420', '5504', '5962', '6171', '7440'] (active 2023-12-07 21:00:00 to 2023-12-07 21:00:00) vs roster ['1003', '295', '3397', '4710', '5057'] (active 2023-12-08 16:10:00 to 2023-12-10 13:55:00) (0-day gap)
- roster ['5420', '5504', '5962', '6171', '7440'] (active 2023-12-07 21:00:00 to 2023-12-07 21:00:00) vs roster ['1003', '295', '3397', '4710', '965'] (active 2024-01-18 18:00:00 to 2024-01-20 18:00:00) (41-day gap)
- roster ['5420', '5504', '5962', '6171', '7440'] (active 2023-12-07 21:00:00 to 2023-12-07 21:00:00) vs roster ['1003', '295', '3397', '4710', '965'] (active 2024-01-31 19:55:00 to 2025-06-19 22:00:00) (54-day gap)

**Assessment**: (a) same organization with roster turnover
**Proposed decision**: `KEEP_AS_SINGLE_TEAM`
**Rationale**: 210 matches under this name with clean sequential roster eras (each era's appearances are chronologically disjoint from the next - a roster hand-off pattern, not simultaneous usage). Per the review's own rule, zero overlap between two eras of an established, high-volume name is NOT by itself evidence of a different organization - full roster turnover over a multi-year dataset is expected for real orgs. Recommend keeping as a single identity, with the era boundaries available for anyone who later wants roster-level (not org-level) granularity.

## `Team StRoGo`

- First appearance: 2025-12-19 16:35:00  |  Last appearance: 2026-05-16 11:00:00
- Matches: 8  |  Tournaments (3): ['BetBoom Streamers Battle x Динамо CS #4', 'PGL Astana 2026', 'Streamers League #1']
- Generic-name pattern match: False

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['26789', '3059'] | 2025-12-19 16:35:00 | 2025-12-22 22:45:00 | ['283349', '283351', '283358', '283360'] |
| 2 | ['26789', '8306'] | 2026-04-20 18:00:00 | 2026-04-23 20:00:00 | ['291228', '291248', '291234'] |
| 3 | ['37', '612', '6693'] | 2026-05-16 11:00:00 | 2026-05-16 11:00:00 | ['293239'] |

Eras chronologically sequential (no overlap between different rosters' active windows): True

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- roster ['26789', '8306'] (active 2026-04-20 18:00:00 to 2026-04-23 20:00:00) vs roster ['37', '612', '6693'] (active 2026-05-16 11:00:00 to 2026-05-16 11:00:00) (22-day gap)

**Assessment**: (c) insufficient evidence
**Proposed decision**: `MANUAL_REVIEW`
**Rationale**: Only 8 matches under this name; roster eras are sequential (consistent with turnover) but the sample is too small to be confident this isn't actually two different low-tier squads that happened to reuse a common-sounding name. Needs manual review.

## `mix52`

- First appearance: 2026-01-27 11:00:00  |  Last appearance: 2026-01-28 11:00:00
- Matches: 4  |  Tournaments (1): ['MySkill Pro League Series 1']
- Generic-name pattern match: True

**Chronological roster eras** (consecutive appearances sharing an identical roster; appearances with an unknown/blank roster are omitted from era-clustering):

| era | roster (player_ids) | first seen | last seen | team_ids |
|---|---|---|---|---|
| 1 | ['25998', '26468', '27129', '40'] | 2026-01-27 11:00:00 | 2026-01-28 11:00:00 | ['285679', '285687', '285690', '285693'] |

Eras chronologically sequential (no overlap between different rosters' active windows): n/a (only one roster era)

**Close-in-time zero-roster-overlap events** (era-to-era, gap <= 60 days or overlapping - reported once per distinct-roster pair, not once per match):

- None found at the era level in this deeper, name-specific re-check.

**Assessment**: (b) likely recycled/ambiguous team name
**Proposed decision**: `EXCLUDE_FROM_IDENTITY_DEPENDENT_FEATURES`
**Rationale**: Generic/placeholder-looking name with a low match count - consistent with an ad-hoc qualifier stand-in roster rather than a persistent organization.

## Summary table

| team_name | matches | assessment | proposed decision |
|---|---|---|---|
| Aurora Gaming | 267 | (a) | `KEEP_AS_SINGLE_TEAM` |
| Team 3DMAX | 255 | (a) | `KEEP_AS_SINGLE_TEAM` |
| Heroic | 235 | (a) | `KEEP_AS_SINGLE_TEAM` |
| Team Spirit | 210 | (a) | `KEEP_AS_SINGLE_TEAM` |
| 9INE | 206 | (a) | `KEEP_AS_SINGLE_TEAM` |
| Sashi Esport | 198 | (a) | `KEEP_AS_SINGLE_TEAM` |
| ECSTATIC | 183 | (a) | `KEEP_AS_SINGLE_TEAM` |
| Ninjas in Pyjamas | 182 | (a) | `KEEP_AS_SINGLE_TEAM` |
| AMKAL Esports | 126 | (a) | `KEEP_AS_SINGLE_TEAM` |
| MOUZ NXT | 106 | (a) | `KEEP_AS_SINGLE_TEAM` |
| SPARTA Esports | 66 | (a) | `KEEP_AS_SINGLE_TEAM` |
| Team Shadowkek | 36 | (c) | `MANUAL_REVIEW` |
| Rhyno Esports | 33 | (c) | `MANUAL_REVIEW` |
| ENTERPRISE esports | 29 | (c) | `MANUAL_REVIEW` |
| Team Buster | 26 | (c) | `MANUAL_REVIEW` |
| Team CS2NEWS | 25 | (c) | `MANUAL_REVIEW` |
| BIG EQUIPA | 19 | (c) | `MANUAL_REVIEW` |
| Entropiq | 15 | (c) | `MANUAL_REVIEW` |
| MANA eSports | 10 | (c) | `MANUAL_REVIEW` |
| Team StRoGo | 8 | (c) | `MANUAL_REVIEW` |
| BRUTE | 7 | (c) | `MANUAL_REVIEW` |
| mix52 | 4 | (b) | `EXCLUDE_FROM_IDENTITY_DEPENDENT_FEATURES` |
| For The Win eSports | 3 | (c) | `MANUAL_REVIEW` |
| Team 2S | 2 | (b) | `EXCLUDE_FROM_IDENTITY_DEPENDENT_FEATURES` |
| Team 7AM | 1 | (b) | `EXCLUDE_FROM_IDENTITY_DEPENDENT_FEATURES` |

Proposed-decision counts: {'KEEP_AS_SINGLE_TEAM': 11, 'MANUAL_REVIEW': 11, 'EXCLUDE_FROM_IDENTITY_DEPENDENT_FEATURES': 3}

**None of these proposed decisions have been applied.** `data/interim/team_aliases.csv` is unchanged by this script. Applying any of them (splitting an identity into eras, excluding a name from identity-dependent features, etc.) is a Phase 3+ decision requiring human sign-off.
