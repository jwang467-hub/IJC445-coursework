# load data Packages 
library(tidyverse)
library(tidytext)
library(textclean)
library(scales)

# read dataset
df <- read_csv("billboard_24years_lyrics_spotify.csv")

# add a unique id + decade group
df <- df %>%
  mutate(
    song_id = row_number(),
    decade = case_when(
      year >= 2000 & year < 2010 ~ "2000s",
      year >= 2010 & year < 2020 ~ "2010s",
      year >= 2020              ~ "2020s",
      TRUE ~ NA_character_
    )
  )

# clean lyrics text (basic cleaning for more reliable word counting) 
df <- df %>%
  mutate(
    lyrics = str_replace_all(lyrics, "\\[.*?\\]", ""),
    lyrics = replace_contraction(lyrics),
    lyrics = replace_non_ascii(lyrics)
  )

# tokenise lyrics into words (turn long text into one-word-per-row format)
df_words <- df %>%
  unnest_tokens(word, lyrics)

# compute basic lyric structure features per song
word_summary <- df_words %>%
  group_by(song_id) %>%
  summarise(
    word_count   = n(),
    unique_words = n_distinct(word),
    .groups = "drop"
  )

# sentiment analysis using Bing lexicon (positive / negative words)
bing <- get_sentiments("bing")

sentiment_counts <- df_words %>%
  inner_join(bing, by = "word", relationship = "many-to-many") %>%
  group_by(song_id) %>%
  summarise(
    positive = sum(sentiment == "positive"),
    negative = sum(sentiment == "negative"),
    .groups = "drop"
  )

# combine word features + sentiment features into one table (per song) 
song_summary <- word_summary %>%
  left_join(sentiment_counts, by = "song_id") %>%
  replace_na(list(positive = 0, negative = 0)) %>%
  mutate(
    sentiment_score = positive - negative,
    positive_ratio  = positive / word_count,
    negative_ratio  = negative / word_count,
    unique_ratio    = unique_words / word_count
  ) %>%
  filter(word_count > 0) %>%
  left_join(df %>% select(song_id, year, decade), by = "song_id")

# Figure 1 
avg_word_by_year <- song_summary %>%
  group_by(year) %>%
  summarise(avg_word_count = mean(word_count), .groups = "drop")

p1 <- ggplot(avg_word_by_year, aes(x = year, y = avg_word_count)) +
  geom_line(color = "#2C7FB8", linewidth = 1) +
  geom_point(color = "#2C7FB8", size = 2) +
  labs(
    title = "Average Lyrics Length of Billboard Hot 100 Songs (2000–2023)",
    x = "Year",
    y = "Average lyrics length (word count)"
  ) +
  theme_classic(base_size = 12)


print(p1)

# Figure 2 (boxplot) 
p2_box <- ggplot(song_summary, aes(x = decade, y = sentiment_score, fill = decade)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c(
    "2000s" = "#A6CEE3",
    "2010s" = "#B2DF8A",
    "2020s" = "#FDBF6F"
  )) +
  labs(
    title = "Distribution of Sentiment Scores of Song Lyrics by Decade",
    x = "Decade",
    y = "Sentiment score"
  ) +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "none"
  )


print(p2_box)

# Figure 3 (scatter + loess) 
p3 <- ggplot(song_summary, aes(x = word_count, y = sentiment_score, color = decade)) +
  geom_point(alpha = 0.6) +
  geom_smooth(se = FALSE, method = "loess") +
  scale_x_continuous(labels = comma) +
  labs(
    title = "Lyrics length and sentiment score across different decades",
    x = "Lyrics length (word count)",
    y = "Sentiment score",
    color = "Decade"
  ) +
  theme_minimal(base_size = 12)


print(p3)

# Figure 4 (PCA) 
pca_data <- song_summary %>%
  select(word_count, sentiment_score, positive_ratio, negative_ratio, unique_ratio)

pca_result <- prcomp(pca_data, scale. = TRUE, center = TRUE)

# variance explained (for axis labels)
pca_var <- pca_result$sdev^2
pca_var_percent <- pca_var / sum(pca_var)
pc1_var <- round(pca_var_percent[1] * 100, 1)
pc2_var <- round(pca_var_percent[2] * 100, 1)

# PCA coordinates + decade labels
pca_scores <- as_tibble(pca_result$x) %>%
  bind_cols(decade = song_summary$decade)

p4 <- ggplot(pca_scores, aes(x = PC1, y = PC2, color = decade)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(
    title = "PCA of Lyrics Features by Decade",
    x = paste0("PC1 (", pc1_var, "% variance explained)"),
    y = paste0("PC2 (", pc2_var, "% variance explained)"),
    color = "Decade"
  ) +
  theme_minimal(base_size = 12)

print(p4)

