package it.giacomobergami.unibz;

import java.io.File;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CrawlBenchmarkDataset {

    public static void parallel(String path) {
        ExecutorService service = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        Arrays.stream(Objects.requireNonNull(new File(path).listFiles(x -> x.isDirectory() && x.getName().endsWith("_results")))).<Runnable>map(f -> () -> {
            if (f.getName().endsWith("_results")) {
                try (CrawlResultsFolder folder = new CrawlResultsFolder(f.getAbsoluteFile().toString())) {
                    System.out.println("[Sequential] Extending experiments for: "+f.getAbsoluteFile().toString());
                    folder.dump();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }).forEach(service::submit);
    }

    public static void sequential(String path) {
        //ExecutorService service = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        Arrays.stream(Objects.requireNonNull(new File(path).listFiles(x -> x.isDirectory() && x.getName().endsWith("_results"))))/*.<Runnable>map(f -> () -> {
            try (CrawlResultsFolder folder = new CrawlResultsFolder(f.getAbsoluteFile().toString())) {
                System.out.println("[Parallel] Extending experiments for: "+f.getAbsoluteFile().toString());
                folder.dump();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).forEach(service::submit);*/
                .forEach(f ->  {
                    if (f.getName().endsWith("_results")) {
                        try (CrawlResultsFolder folder = new CrawlResultsFolder(f.getAbsoluteFile().toString())) {
                            System.out.println("[Sequential] Extending experiments for: "+f.getAbsoluteFile().toString());
                            folder.dump();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                });
    }

    public static void main(String args[]) {
        parallel(args[0]); ///home/giacomo/PycharmProjects/dmm2/data/experiments
    }

}
