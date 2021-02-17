package it.giacomobergami.unibz;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class CrawlBenchmarkDataset {

    public static void parallel(String path) throws InterruptedException {
        runFilteredParallel(path, "xray");
        //runFilteredParallel(path, "bpi11");
        //runFilteredParallel(path, "sepsis");
    }

    public static void sequential(String path) {
        runFilteredSequential(path, "xray");
        runFilteredSequential(path, "bpi11");
        runFilteredSequential(path, "sepsis");
    }

    private static void runFilteredSequential(String path, String bpi11) {
        Arrays.stream(Objects.requireNonNull(new File(path).listFiles(x -> x.isDirectory() && x.getName().endsWith("_results"))))
                .filter(f -> f.getName().startsWith(bpi11))
                .sorted()
                .forEach(f -> {
                    if (f.getName().endsWith("_results")) {
                        try (CrawlResultsFolder folder = new CrawlResultsFolder(f.getAbsoluteFile().toString())) {
                            System.out.println("[Sequential] Extending experiments for: " + f.getAbsoluteFile().toString());
                            folder.dump();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                    System.gc();
                });
    }

    private static void runFilteredParallel(String path, String bpi11) throws InterruptedException {
        //ExecutorService service = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        List<File> files = Arrays.stream(Objects.requireNonNull(new File(path).listFiles(x -> x.isDirectory() && x.getName().endsWith("_results"))))
                .filter(f -> f.getName().startsWith(bpi11))
                .collect(Collectors.toList());
        files.parallelStream().<Runnable>forEach(f -> {
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

    public static void main(String args[]) throws InterruptedException {
        runFilteredSequential(args[0], args[1]);
    }

}
