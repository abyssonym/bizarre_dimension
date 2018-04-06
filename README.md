# Bizarre Dimension – EarthBound Ancient Cave Randomizer

Bizarre Dimension is a program that randomizes a ROM for the Super Nintendo game EarthBound, providing endless unique gameplay experiences with many distinct modes.

Windows users can simply download the `eb_bizarre_dimension.exe` file and run it; other platforms can run `randomizer.py` with Python 2. You will need to provide a ROM file of EarthBound; we will not provide this file. When you run the program, it will first ask for the location of this file; simply put the file in the same directory as the program and enter the filename.

Next, the program will ask for a seed. Simply leave this field blank to get a random seed, or enter a specific integer if you want to play the same generated game as a friend. Any ROMs generated with the same version, seed, and flags will be identical.

Next, the program will ask you which flags you would like to set. You can pick and choose which features of the randomizer you want enabled, or you can simply enter a blank line to get the default, all features enabled. After all options are selected, a new ROM file will be created, as well as a `.spoiler.json` file. If you get stuck during the game, you can use this file at [https://eb.compnode.net](https://eb.compnode.net) to get a visual map of where doors lead to, what chest contents are, and other spoiler information.

By using `Help!` on the ATM card, you will be able to get details on the version of the randomizer used, the flags selected, and the seed of the particular game.

Most flags are self-explanatory, but additional details for some modes are included below.

## Ancient Cave mode – `a`

Ancient Cave mode completely changes how the game is played. Instead of proceeding through the normal storyline of EarthBound, all rooms and doors have been shuffled around into a multi-level maze. You start with all four party members, and your goal is to proceed through all 8 levels of the maze, each level guarded by a Shiny Spot, and reach and defeat Giygas. Enemy spawn locations grow in difficulty as you proceed deeper into the maze.

You will not have to do any storyline events, like riding the Sky Runner or beating Carpainter, to beat the game in this mode. Often, you can do story events to "skip" deeper into the cave. Use common sense when taking skips, as some may break the game; giving an item to a monkey in Monkey Cave is safe, for example, while riding the Sky Runner is almost definitely not.

## Keysanity mode – `k`

**WARNING**: This mode is currently in beta test, and may not fully work correctly. Any feedback on problems encountered would be helpful.

Keysanity mode also radically changes how the game is played. 14 different key items have been shuffled around throughout the world; Mayor Pirkle may give you the Bicycle, while the Bike Shop guy may give you the Carrot key. To help you on this more complicated quest, however, Ness already knows PSI Teleport, and all available teleport locations are unlocked at the start of the game (including a bonus teleport to South Winters). Your goal is to beat the game as normal, but getting to all 8 Your Sanctuary locations will be more of a challenge.

The list of items that have had their locations shuffled in this mode is as follows:
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

Because you can get a Jar of Fly Honey through one of these 14 locations, it is not necessary to do the Jeff-alone-in-Winters part of the storyline. However, you can still do so if you wish, as the Boogey Tent will still contain a Jar of Fly Honey as well.

Be careful with how you proceed through storyline events! If you take a very unusual order, it is possible you may lock yourself out of having a character available until you complete Magicant. Since you can teleport anywhere instantly, you can get the game into very unusual states; this is expected and encouraged to take advantage of to get a lower time.

Keysanity mode is incompatible with Ancient Cave mode, and if both modes are selected, Keysanity will be disabled. (Therefore, if you simply enter a blank line at flag selection, you will be playing Ancient Cave, not Keysanity.)

## Gift box contents – `g`

Randomizing gift box contents works differently in Ancient Cave mode and in non-Ancient Cave mode. In Ancient Cave mode, gift boxes will (typically) increase in value as you proceed deeper into the cave. Almost anything could be in one of the gift boxes, including a key item (such as the Pencil Eraser) that may allow you to perform skips at various points in the cave.

In non-Ancient Cave mode, gift boxes will be replaced with an item that is similar in value to their contents in the normal game. However, every gift box has a 20% chance of being replaced by an item of any value, so there is a chance you could get a very good item very early.

## Run button patch

Bizarre Dimension includes a patch that adds a run button to the game. Holding `Y` will make the characters run as if they had used a Skip Sandwich. If you do not want this feature, follow the instructions in the program to disable it.

## Special codes

There are a few special codes you can enter along with the flags. For example, to add `wildgifts` and `funsize` to a game with the `a`, `g`, and `p` flags only set, you would enter `agpwildgiftsfunsize`.

- `wildgifts` – In Non-Ancient Cave mode, with gift box randomization, 20% chance of any item increased to 100% chance.
- `funsize` – With PC Sprite randomization, additional PC sprites that don't conform to normal PC size are added to the pool of options. These sprites occasionally cause small graphical hiccups, but they can add a lot of variety.
-  `devmode` – Additional details are added to the `.spoiler.json` file, usually only useful for development purposes.

## Known issues

- Occasionally, dying can cause the screen to go black and not return, while you can hear music and sound effects as normal. Regularly save so you can reload if this occurs. Also, you can try to walk around and hope to walk through a door to reset the screen. More information about this is needed to debug it.
- More instances of an equippable item than can be used can be generated.
- Possible Lost World "geyser softlock" in AC mode. More information is needed.

## Contact

Bizarre Dimension was originally created by Abyssonym; you can find him at [Twitter](https://www.twitter.com/abyssonym) or [his website](http://www.abyssonym.com/).

Current development on this fork is being led by pickfifteen; you can contact him either via the Issue Tracker on this repo, [Twitter](https://www.twitter.com/pickfifteen), or on the [EarthBound Speedrunning Discord](https://discord.gg/WWVYwkE).
