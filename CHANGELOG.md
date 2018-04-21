# Changelog

## v12 (In Progress)
- PC sprites: New method of randomizing allows for much wider variety of sprites to appear
### Bugfixes
- Ancient Cave: Logic errors that caused occasional incorrect enemies to appear in Electro Specter cave areas fixed.

## v11 (2018-04-10)
- Keysanity: Now features a PSI Teleport destination of North Onett. This allows access to north locations if you teleport before clearing certain flags.
- Keysanity: Suporma location added to distribution pool.
- Keysanity: Meteorite piece added to items pool (Suporma will not appear in pool).
### Bugfixes
- NPC sprites: Hang at Buzz Buzz will no longer occur.
- Keysanity: Venus will always give you her item.
- Keysanity: Pyramid can be accessed without fighting Kraken.

## v10 (2018-04-05)
- Keysanity: Now features a PSI Teleport destination of South Winters. This allows much easier routing and can actually avoid the Jeff-alone events now.
### Bugfixes
- Gift box contents: Dummy item named "Null" should no longer appear in randomized chests.
- Keysanity: Gerard Montague always appears outside of the mine, so you cannot get locked out of Diamond location.
- Keysanity: Mr Spoon can always request the autograph, so you cannot get locked out of Signed banana location.
- Keysanity: Bubble Monkey should always be available on the north shore after he runs off the first time.
- Keysanity: Various impossible item routings no longer will occur.

## v9 (2018-03-31)
- Keysanity mode added. Certain key items have been mixed up, but all teleports are available from the start of the game!
- Here are the item locations that have been shuffled in Keysanity mode (OUTDATED):
  - Franklin badge
  - Shyness book
  - King banana
  - Key to the shack
  - Hawk eye
  - Bicycle
  - Wad of bills
  - Diamond
  - Signed banana
  - Pencil eraser
  - Key to the tower
  - Town map
  - Carrot key
  - Tendakraut - but the Tendakraut has been transformed into a Jar of Fly Honey!
- Because there is a Jar of Fly Honey in the item pool, proceeding through Winters as Jeff is not required; however, if you choose to, the Bubble Monkey rope event has been patched to work even if you have extra companions.
- Keysanity mode is incompatible with Ancient Cave mode, and probably needs a lot of testing. Let me know if you encounter issues.

## v8 (2018-03-26)
- Non-essential dialogs shuffle with the 'd' flag (with special surprises possible at Pokey).
- Chests in non-AC mode have a 20% chance to have any non-key item.
- Use the code 'wildgifts' in your flags to make that chance 100%.
- Money chests in non-AC mode randomize to (low-cost) items.
- Chests that were not properly randomized in AC mode fixed: 
  - Tracy's room
  - Bracer of kings room
  - Cold remedy room (Giant Step)
  - Bag of dragonite room (Pyramid)
  - Magicant red area

## v7 (2018-03-21)
- Sanctuary Boss rooms in AC now always go front-to-back as they do in the base game.
- Non-Euclidian room layouts at Electro Specter no longer happen.
- Spoiler site at http://eb.compnode.net/ now displays chest locations and contents.
- Spoiler site at http://eb.compnode.net/ now displays boss locations and enemies.
- Odd-shaped PCs no longer appear in PC Sprite randomization by default due to graphical glitches.
- Use the code 'funsize' in your flags to get the odd-shaped PCs back.

## v6 (2018-03-19)
- Program now outputs a spoiler file in AC mode.
- Use this file at http://eb.compnode.net/ to view a map of doors and destinations. (More features coming soon.)

## v5 (2018-03-18)
- "Help" on ATM Card displays randomizer version info, seed, flags chosen.
- Mom's state set to heal you as normal.
- Phone in Ness's house no longer removes existing phone numbers.
- Non-gift-box NPCs no longer receive gift-box sprites.
- Chaos Theater show skipped if 'a' and 'p' flags selected due to panning bug.

## v4 (2018-03-09)
- Topolla Theater routing bug fixed.

## v3 (2018-03-09)
- NPC Sprite Swap and PC Sprite Swap flags added.
- Run button optional patch added.
- Enemy PP removed from intershuffle (fewer enemies will have no PP for their PP attacks).

## v2 (2018-03-05)
- Battle backgrounds randomized.
- Enemy battle sprite pallettes randomized.
- Hotels fixed.
- Phones in Winters locations enabled.
- Belch removed from Sanctuary Spot boss possibilities.

## v1 (2018-01-22)
- Praying at Giygas no longer softlocks.
- Exit Mice are now functional. They will take you back to the most recent Sanctuary Spot you visited, or home if you have not visited any.
- Sanctuary Spots are working now.
- Many items will now appear only once per dungeon, notably items that can only be equipped by one character.
- Many items that were unsellable/undroppable now can be.
- Added various protections against the enemy despawn bug.
- Ness starts with the Escargo Express phone number memorized.
- Sanctuary bosses now report their proper order.
- Some hotspots and other events have been modified, such as Poo leaving after the Pyramid.
- NPCs that start battles should never appear in the first zone.
- Various fixes to routing.