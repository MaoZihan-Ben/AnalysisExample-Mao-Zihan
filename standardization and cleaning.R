
library(tidyverse)    
library(naniar)       
library(here)         


input_folder <- "C:/Users/30444/Desktop/data"   # 原始資料
output_folder <- file.path(input_folder, "cleaned data")
dir.create(output_folder, showWarnings = FALSE)
log_file <- file.path(output_folder, "清洗紀錄表.csv") # 清洗日誌

# 問卷常見特殊缺失編碼，RCT調研資料經常使用-99、-88代表拒答/遺漏
special_missing_code <- c(-99, -88, "", " ", "NA")

# 初始化清洗日誌，紀錄每個檔案處理前後行數
log_df <- tibble(
  檔名 = character(),
  原始行數 = integer(),
  刪除缺失行數 = integer(),
  刪除重複行數 = integer(),
  清洗後行數 = integer(),
  處理時間 = Sys.time()
)

# 缺失值處理 
# 輸出缺失統計 缺失熱圖 選擇刪除含缺失列
handle_missing_values <- function(df, filename){
  original_rows <- nrow(df)
  
  df <- df %>% replace_with_na_all(condition = ~.x %in% special_missing_code)
  
  cat("\n===== 處理【",filename,"】缺失值 =====\n")
  cat("各欄位缺失數量統計：\n")
  print(summarise_all(df, ~sum(is.na(.))))
  
  # 繪缺失熱圖
  plot_path <- file.path(output_folder, str_c(str_remove(filename,".csv"),"_缺失熱圖.png"))
  ggsave(plot_path, vis_miss(df, show_perc = TRUE), width =12, height =6)
  cat("缺失熱圖已儲存：",plot_path,"\n")
  
  # 刪除含有NA的整行
  df_dropped <- drop_na(df)
  missing_rows <- original_rows - nrow(df_dropped)
  
  cat("已刪除", missing_rows, "筆含有缺失的觀測\n")
  cat("缺失處理後剩餘", nrow(df_dropped), "筆資料\n")
  
  return(list(data = df_dropped, missing_del = missing_rows))
}

#重複值處理函數區塊 
# 兩種模式：(1)全部欄位完全相同 (2)依據ID欄位檢查主鍵重複
handle_duplicates <- function(df, filename, id_col = NULL){
  original_rows <- nrow(df)
  cat("\n===== 處理【",filename,"】重複值 =====\n")
  
  if(!is.null(id_col) && id_col %in% colnames(df)){
    dup_id <- df[[id_col]][duplicated(df[[id_col]])]
    duplicate_rows <- length(dup_id)
    df_clean <- df %>% distinct(.data[[id_col]], .keep_all = TRUE)
    cat("使用ID欄位：",id_col,"檢查重複\n")
  }else{
    df_clean <- distinct(df)
    duplicate_rows <- original_rows - nrow(df_clean)
  }
  
  if(duplicate_rows >0){
    cat("偵測到", duplicate_rows, "筆重複觀測，已移除\n")
  }else{
    cat("未偵測重複資料\n")
  }
  cat("去重後剩餘", nrow(df_clean), "筆資料\n")
  
  return(list(data = df_clean, dup_del = duplicate_rows))
}

#單一檔案處理主函數區
process_single_file <- function(filepath, filename){
  cat("\n\n########################################\n")
  cat("===== 開始處理檔案：",filename," =====\n")
    
  df <- read_csv(filepath, show_col_types = FALSE, locale = locale(encoding = "UTF‑8‑SIG"))
  ori_n <- nrow(df)
  cat("原始資料行數：",ori_n,"\n")
  miss_result <- handle_missing_values(df, filename)
  df_after_miss <- miss_result$data
  del_miss <- miss_result$missing_del
  
  # 重複值處理，填入你的ID欄位名稱
  dup_result <- handle_duplicates(df_after_miss, filename, id_col = NULL)
  df_final <- dup_result$data
  del_dup <- dup_result$dup_del
  
  clean_name <- str_c("cleaned_", filename)
  clean_full_path <- file.path(output_folder, clean_name)
  write_csv(df_final, clean_full_path)
  cat("\n清洗完成！檔案儲存位置：",clean_full_path,"\n")
  
  #寫入日誌
  return(tibble(
    檔名 = filename,
    原始行數 = ori_n,
    刪除缺失行數 = del_miss,
    刪除重複行數 = del_dup,
    清洗後行數 = nrow(df_final),
    處理時間 = Sys.time()
  ))
}


# 讀取資料夾內所有csv，跳過已經cleaned_開頭的清洗檔
file_list <- list.files(input_folder, pattern = "\\.csv$")
file_list <- file_list[!str_starts(file_list,"cleaned_")]

cat("===== 開始批量處理資料夾：",input_folder,"=====\n")
for(f in file_list){
  full_path <- file.path(input_folder, f)
  tryCatch({
    one_log <- process_single_file(full_path, f)
    log_df <- bind_rows(log_df, one_log)
  }, error = function(e){
    cat("\n❌處理檔案",f,"發生錯誤：",e$message,"\n")
  })
}


write_csv(log_df, log_file)
cat("\n===== 全部檔案處理完畢！=====\n")
cat("完整清洗紀錄已儲存至：",log_file,"\n")
print(log_df)
