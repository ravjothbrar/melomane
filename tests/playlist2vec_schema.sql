-- MySQL dump 10.13  Distrib 5.7.33, for Linux (x86_64)
--
-- Host: localhost    Database: spotifydb
-- ------------------------------------------------------
-- Server version	5.7.33-0ubuntu0.18.04.1

/*!40101 SET NAMES utf8 */;

DROP TABLE IF EXISTS `album`;
CREATE TABLE `album` (
  `id` varchar(100) NOT NULL,
  `name` varchar(500) DEFAULT NULL,
  `uri` varchar(1000) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `artist`;
CREATE TABLE `artist` (
  `id` varchar(100) NOT NULL,
  `name` varchar(500) DEFAULT NULL,
  `uri` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `playlist`;
CREATE TABLE `playlist` (
  `id` varchar(100) NOT NULL,
  `name` varchar(500) NOT NULL,
  `followers` int(11) DEFAULT NULL,
  `uri` varchar(1000) DEFAULT NULL,
  `total_tracks` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `p_name_idx` (`name`),
  KEY `p_total_tracks_idx` (`total_tracks`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `track`;
CREATE TABLE `track` (
  `id` varchar(100) NOT NULL,
  `name` varchar(500) DEFAULT NULL,
  `duration` int(11) DEFAULT NULL,
  `popularity` double DEFAULT NULL,
  `explicit` varchar(10) DEFAULT NULL,
  `preview_url` varchar(10000) DEFAULT NULL,
  `uri` varchar(1000) DEFAULT NULL,
  `album_id` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `album_id_idx` (`album_id`),
  CONSTRAINT `album_id` FOREIGN KEY (`album_id`) REFERENCES `album` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `track_artist1`;
CREATE TABLE `track_artist1` (
  `track_id` varchar(100) DEFAULT NULL,
  `artist_id` varchar(100) DEFAULT NULL,
  UNIQUE KEY `unique_index` (`track_id`,`artist_id`),
  KEY `track_id_idx` (`track_id`),
  KEY `artist_id_idx` (`artist_id`),
  CONSTRAINT `ta_artist_id1` FOREIGN KEY (`artist_id`) REFERENCES `artist` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `ta_track_id1` FOREIGN KEY (`track_id`) REFERENCES `track` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `track_playlist1`;
CREATE TABLE `track_playlist1` (
  `track_id` varchar(100) DEFAULT NULL,
  `playlist_id` varchar(100) DEFAULT NULL,
  UNIQUE KEY `unique_index` (`track_id`,`playlist_id`),
  KEY `track_id_idx` (`track_id`),
  KEY `playlist_id_idx` (`playlist_id`),
  CONSTRAINT `tp_playlist_id1` FOREIGN KEY (`playlist_id`) REFERENCES `playlist` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `tp_track_id1` FOREIGN KEY (`track_id`) REFERENCES `track` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Dump completed on 2021-02-11 15:42:39
