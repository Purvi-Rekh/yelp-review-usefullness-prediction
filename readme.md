{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Yelp review usefullness prediction\n",
    "\n",
    "## By Purvi Joshi\n",
    "\n",
    "# Overview : \n",
    "\n",
    "Yelp has become very useful app now a days to serach for almost all utility. This project is about predicting if yelp review will be useful or not. Generally people read only first few review so if we can successfully predict if people will find a review useful, yelp can give it a priority and put in before reviews that are predicted as not useful.\n",
    "\n",
    "\n",
    "[![N|Solid](https://images.app.goo.gl/2RpbrthcxcdqFMgS6)](https://nodesource.com/products/nsolid)\n",
    "\n",
    "# Dataset - \n",
    "\n",
    "[Dataset is available here :](https://www.yelp.com/dataset/challenge)\n",
    "This dataset contains 6668738 reiews from 179974 different businesses. So if we consider useful count 0 as not useful review and useful count >=1 as useful review than distribution of useful and not useful review out of 6668738 reviews is as below:\n",
    "\n",
    "![Image description](image/total_business_usefulness_review.jpg)\n",
    "\n",
    "\n",
    "For this project from 1489 indian restaurants in USA, 79767 reviews are extracted from json file using pypark. So if we consider useful count 0 as not useful review and useful count >=1 as useful review than distribution of useful and not useful review out of 79767 reviews is as below:\n",
    "\n",
    "\n",
    "![Image description](image/indian_resto_usefulness_review.jpg)\n",
    "\n",
    "\n",
    "Data are taken from 2 files review.json and business.json, following columns are used for this project:\n",
    "\n",
    "review file columns\n",
    "\n",
    "        Column - Description - Datatype\n",
    "        \n",
    "  - business_id - company id - alphanumeric\n",
    "  - text - comments made by reviwer - Text statements\n",
    "  - useful - if comment was helpful or not, liked by reviwers - number\n",
    "  - funny - if comment was funny or not, liked as funny by reviwers - number\n",
    "  - cool - if comment was cool or not, liked as cool comment by reviwers - number\n",
    "  \n",
    "business file columns\n",
    "\n",
    "        Column - Description - Datatype\n",
    "        \n",
    "  - business_id - company id - alphanumeric\n",
    "  - name - Name of company - String\n",
    "  - categories - Type of business - String\n",
    "\n",
    "# EDA\n",
    "\n",
    "# Data Cleaning\n",
    "- Filling missing values in pros and cons\n",
    "- Convert pros and cons from object data type to string\n",
    "    \n",
    "# Feature selection\n",
    "##### Performed following steps to select most frequent words from pros and cons:\n",
    "- Normalized text \n",
    "- Remove special charcters using Regular Expression\n",
    "- Tokenization\n",
    "- Remove stop words\n",
    "- create vector using tf-idf\n",
    "\n",
    "# Number of words and number of reviews distribution\n",
    "\n",
    "![Image description](image/thresold6.jpg)\n",
    "\n",
    "\n",
    "# Modle selection and ROC curve\n",
    "\n",
    "Using over and under sampling\n",
    "\n",
    "![Image description](image/roc_curve_tfidf_thresold_3.png)\n",
    "\n",
    "Using 2-gram \n",
    "\n",
    "![Image description](image/2_gram_roc_useful_cool_funny_feature1000.jpg)\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "   \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
