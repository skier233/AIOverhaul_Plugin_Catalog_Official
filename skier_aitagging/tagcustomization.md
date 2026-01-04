# Customizing or Removing Tags
You may find that you don't care about some of the tags that the AI produces and want to remove them or tune when they are present. This is easily achievable through a few configuration files in the AI Server.

# Configuring Tags
1. Navigate to ./plugins/skier_aitagging/tag_settings.csv
2. Open the csv file in an editor like Excel on Windows or a Linux equivalent such as Modern CSV
## Editing the CSV File
Within this csv file, you'll see headers at the top row. Here is a description of what each of these columns is for:
|tag_name|stash_Name |markers_enabled |scene_tag_enabled |image_enabled |min_marker_duration |max_gap|
|----------|-----------|------------|---------|-------|----|
|DO NOT EDIT|Name of the tag that will be shown in stash (by default it will be tag_name_AI) |Whether markers will be created for this tag |Whether scene tags will be created for this tag |Whether image tags will be created for this tag |Required time this must be present in the scene to be applied as a scene tag (seconds is default but % is also allowed)| Markers shorter than this length will not be created for this tag|Raising this will make markers merge together more often leading to longer markers. Lowering it will prefer smaller more technical precise markers.
